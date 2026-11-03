//! Shared-word Monte Carlo producer for diffusion features G07-G10.

use super::{MAX_MC_WORKING_BYTES, ProfileCache, unavailable};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureValue};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;
use std::sync::Arc;

pub(super) struct MonteCarloProducer {
    spec: FeatureSpec,
    cache: Arc<ProfileCache>,
}

impl FeatureProducer for MonteCarloProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        if n > self.cache.config.max_n as usize {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "diffusion max_n={} exceeded by n={n}",
                    self.cache.config.max_n
                ),
            );
        }
        let Some(pair_count) = n.checked_mul(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo pair count overflow".into(),
            );
        };
        let Some(working_bytes) = pair_count
            .checked_mul(self.cache.config.mc_samples)
            .and_then(|value| value.checked_mul(8 * 8))
        else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo working-set size overflow".into(),
            );
        };
        if working_bytes > MAX_MC_WORKING_BYTES {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "Monte Carlo working set {working_bytes} bytes exceeds {MAX_MC_WORKING_BYTES}"
                ),
            );
        }
        let Some(moments) = self.cache.get_mc(state, policy.seed) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo sampling exceeded a checked allocation or numeric budget".into(),
            );
        };

        let channels = self.spec.channels;
        let output_cells = if self.spec.id.ordinal() == 7 {
            n
        } else {
            pair_count
        };
        let Some(value_count) = output_cells.checked_mul(channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo output allocation failed".into(),
            );
        }

        match self.spec.id.ordinal() {
            7 => values.extend(moments.mean_weight.iter().map(|value| *value as f32)),
            8 => {
                for index in 0..pair_count {
                    values.extend([
                        moments.covariance_weight[index] as f32,
                        moments.mean_intersection[index] as f32,
                        moments.mean_xor[index] as f32,
                        moments.covariance_weight_intersection[index] as f32,
                        moments.covariance_weight_xor[index] as f32,
                    ]);
                }
            }
            9 => {
                for index in 0..pair_count {
                    values.extend([
                        moments.mean_character_pair[index] as f32,
                        moments.covariance_character_pair[index] as f32,
                    ]);
                }
            }
            10 => {
                let stationary_off_diagonal = -1.0 / (2.0f64.powi(n as i32) - 1.0);
                let gates_per_sample = moments.total_gates as f64 / moments.samples as f64;
                for first in 0..n {
                    for second in 0..n {
                        let index = first * n + second;
                        let stationary = if first == second {
                            1.0
                        } else {
                            stationary_off_diagonal
                        };
                        values.extend([
                            moments.standard_error_intersection[index] as f32,
                            moments.standard_error_xor[index] as f32,
                            moments.standard_error_character_pair[index] as f32,
                            (moments.mean_character_pair[index] - stationary).abs() as f32,
                            moments.samples as f32,
                            gates_per_sample as f32,
                            self.cache.config.mc_tau as f32,
                        ]);
                    }
                }
            }
            _ => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "Monte Carlo producer received an unregistered feature ID".into(),
                );
            }
        }
        if values.len() != value_count || values.iter().any(|value| !value.is_finite()) {
            return unavailable(
                FeatureStatus::Unsupported,
                "Monte Carlo producer generated a non-finite or malformed output".into(),
            );
        }

        let shape = if self.spec.id.ordinal() == 7 {
            vec![n, channels]
        } else {
            vec![n, n, channels]
        };
        FeatureComputation::Computed(FeatureValue {
            shape,
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl MonteCarloProducer {
    pub(super) fn new(spec: FeatureSpec, cache: Arc<ProfileCache>) -> Self {
        Self { spec, cache }
    }
}
