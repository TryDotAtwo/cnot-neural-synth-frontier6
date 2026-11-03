//! Opt-in source-relative CNOT diffusion features G01-G04 and G06-G10.
//!
//! The source computes finite continuous-time Markov semigroups over exact
//! column-weight and rank-2 pair-histogram classes. Uniformization uses the same transition counts and
//! retains the source's floating-semigroup status; it does not claim a formal
//! numerical error certificate.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{FeatureId, FeatureSpec, feature_spec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::sync::Arc;

mod cache;
mod monte_carlo;
mod profiles;
use cache::ProfileCache;
use monte_carlo::MonteCarloProducer;
use profiles::pair_state_count;

const MAX_CACHED_WIDTHS: usize = 16;
const MAX_MC_CACHE_ENTRIES: usize = 8;
const MAX_MC_WORKING_BYTES: usize = 256 * 1024 * 1024;
pub(super) const MAX_UNIFORMIZATION_TERMS: usize = 1_000_000;
pub(super) const POISSON_TAIL_SIGMAS: f64 = 12.0;
pub(super) const POISSON_TAIL_MARGIN: f64 = 32.0;

#[derive(Clone, Copy, Debug)]
pub struct DiffusionConfig {
    /// Four values are fixed by the canonical 103-feature channel layout.
    pub taus: [f64; 4],
    pub floor: f64,
    pub max_n: u32,
    /// Explicit state-space limit for the rank-2 pair projection G06.
    pub max_pair_states: usize,
    /// Number of shared random CNOT words sampled by G07-G10.
    pub mc_samples: usize,
    /// Source time multiplier; the sampled Poisson time is `n * mc_tau`.
    pub mc_tau: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            taus: [0.25, 1.0, 4.0, 16.0],
            floor: 1e-30,
            max_n: 256,
            max_pair_states: 10_000,
            mc_samples: 256,
            mc_tau: 1.0,
        }
    }
}

/// Register source-relative diffusion producers G01-G04 and G06-G10.
///
/// Registration is opt-in, like the Python source. The provider identities bind
/// emitted values to their numeric parameters and explicit resource budgets.
pub fn register_diffusion(
    registry: &mut FeatureRegistry,
    config: DiffusionConfig,
) -> Result<(), FeatureBuildError> {
    validate_config(config)?;
    let ids = [
        FeatureId::new(b'G', 1).unwrap(),
        FeatureId::new(b'G', 2).unwrap(),
        FeatureId::new(b'G', 3).unwrap(),
        FeatureId::new(b'G', 4).unwrap(),
        FeatureId::new(b'G', 6).unwrap(),
    ];
    let mc_ids = [
        FeatureId::new(b'G', 7).unwrap(),
        FeatureId::new(b'G', 8).unwrap(),
        FeatureId::new(b'G', 9).unwrap(),
        FeatureId::new(b'G', 10).unwrap(),
    ];
    let specs = registry.specs();
    if let Some(id) = ids
        .iter()
        .chain(mc_ids.iter())
        .find(|id| specs[id.index()].producer_version != "unconfigured")
    {
        return Err(FeatureBuildError(format!(
            "{id} already has a configured provider"
        )));
    }

    let version = producer_version(config);
    let cache = Arc::new(ProfileCache::new(config));
    for id in ids {
        let mut spec = feature_spec(id);
        spec.scope = if id.ordinal() == 6 {
            "full_labeled_matrix;histogram_class_mass_not_labeled_endpoint".into()
        } else {
            "full_labeled_matrix;source_relative_uniform_CNOT".into()
        };
        spec.producer_version = version.clone();
        spec.dependencies = if id.ordinal() == 6 {
            vec!["M".into(), "source_relative_rank2_pair_chain".into()]
        } else {
            vec!["M".into(), "source_relative_column_chain".into()]
        };
        spec.exactness = "floating_semigroup".into();
        spec.cost_tier = if id.ordinal() == 6 { "T2" } else { "T1" }.into();
        spec.wire_roles.clear();
        spec.requires_solver_assisted = false;
        registry.register(Box::new(DiffusionProducer {
            spec,
            cache: Arc::clone(&cache),
        }))?;
    }
    let mc_version = mc_producer_version(config);
    for id in mc_ids {
        let mut spec = feature_spec(id);
        spec.scope =
            "full_labeled_matrix;shared_word_MC_joint_moments_empirical_uncertainty".into();
        spec.producer_version = mc_version.clone();
        spec.dependencies = vec!["M".into(), "shared_word_uniform_ordered_CNOT_MC".into()];
        spec.exactness = "monte_carlo".into();
        spec.cost_tier = "T2".into();
        spec.wire_roles.clear();
        spec.requires_solver_assisted = false;
        registry.register(Box::new(MonteCarloProducer::new(spec, Arc::clone(&cache))))?;
    }
    Ok(())
}

fn validate_config(config: DiffusionConfig) -> Result<(), FeatureBuildError> {
    if config.max_n == 0 {
        return Err(FeatureBuildError("diffusion max_n must be positive".into()));
    }
    if config.max_pair_states == 0 {
        return Err(FeatureBuildError(
            "diffusion max_pair_states must be positive".into(),
        ));
    }
    if config.mc_samples < 2 {
        return Err(FeatureBuildError(
            "diffusion mc_samples must be at least two".into(),
        ));
    }
    if !config.mc_tau.is_finite() || config.mc_tau < 0.0 {
        return Err(FeatureBuildError(
            "diffusion mc_tau must be finite and nonnegative".into(),
        ));
    }
    if config.taus.iter().any(|tau| !tau.is_finite() || *tau < 0.0) {
        return Err(FeatureBuildError(
            "diffusion times must be finite and nonnegative".into(),
        ));
    }
    if !config.floor.is_finite() || !(0.0..1.0).contains(&config.floor) {
        return Err(FeatureBuildError(
            "diffusion floor must be finite and strictly between zero and one".into(),
        ));
    }
    Ok(())
}

fn producer_version(config: DiffusionConfig) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/source_relative_diffusion_v2\0");
    for tau in config.taus {
        digest.update(tau.to_bits().to_le_bytes());
    }
    digest.update(config.floor.to_bits().to_le_bytes());
    digest.update(config.max_n.to_le_bytes());
    digest.update((config.max_pair_states as u64).to_le_bytes());
    let hex = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("diffusion_v2:{hex}")
}

fn mc_producer_version(config: DiffusionConfig) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/shared_word_diffusion_monte_carlo_v1\0");
    digest.update(config.max_n.to_le_bytes());
    digest.update((config.mc_samples as u64).to_le_bytes());
    digest.update(config.mc_tau.to_bits().to_le_bytes());
    digest.update(b"numpy_2.2.6_default_rng_pcg64\0");
    let hex = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("diffusion_mc_v1:{hex}")
}

struct DiffusionProducer {
    spec: FeatureSpec,
    cache: Arc<ProfileCache>,
}

impl FeatureProducer for DiffusionProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        if self.spec.id.ordinal() == 6 {
            return self.compute_pair(state, n);
        }
        if n > self.cache.config.max_n as usize {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "column diffusion max_n={} exceeded by n={n}",
                    self.cache.config.max_n
                ),
            );
        }
        let Some(profile) = self.cache.get(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion checked state/iteration budget exceeded".into(),
            );
        };
        let Some(value_count) = n.checked_mul(self.spec.channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion output allocation failed".into(),
            );
        }
        values.resize(value_count, 0.0f32);

        let row_weights = row_weights(state);
        let column_weights = column_weights(state);
        for wire in 0..n {
            let diagonal_bit = matrix_bit(state, wire, wire) as usize;
            let weight = if self.spec.id.ordinal() == 2 {
                row_weights[wire]
            } else {
                column_weights[wire]
            };
            let Some(other_weight) = weight.checked_sub(diagonal_bit) else {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "diagonal bit exceeds selected row/column weight".into(),
                );
            };
            let Some(class) = profile.class_index(diagonal_bit, other_weight) else {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "selected row/column class is outside the diffusion chain".into(),
                );
            };
            let base = wire * self.spec.channels;
            match self.spec.id.ordinal() {
                1 | 2 => {
                    for time_index in 0..4 {
                        let sample = profile.sample(time_index, class);
                        let offset = base + time_index * 7;
                        values[offset] = sample.raw as f32;
                        values[offset + 1] = profile.log_class_size[class] as f32;
                        values[offset + 2] = sample.stationary as f32;
                        values[offset + 3] = (sample.log_mass - sample.stationary.ln()) as f32;
                        values[offset + 4] = f32::from(sample.exact_zero);
                        values[offset + 5] = f32::from(sample.underflow);
                        values[offset + 6] = f32::from(sample.raw < self.cache.config.floor);
                    }
                }
                3 => {
                    for time_index in 0..4 {
                        let sample = profile.sample(time_index, class);
                        let offset = base + time_index * 4;
                        values[offset] = sample.log_mass as f32;
                        values[offset + 1] = f32::from(sample.exact_zero);
                        values[offset + 2] = f32::from(sample.underflow);
                        values[offset + 3] = f32::from(sample.raw < self.cache.config.floor);
                    }
                }
                4 => {
                    for time_index in 0..3 {
                        let current = profile.raw(time_index + 1, class);
                        let previous = profile.raw(time_index, class);
                        values[base + time_index] = (current - previous) as f32;
                    }
                }
                _ => {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "diffusion producer received an unregistered feature ID".into(),
                    );
                }
            }
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, self.spec.channels],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl DiffusionProducer {
    fn compute_pair(&self, state: &MatrixState, n: usize) -> FeatureComputation {
        if n < 2 {
            return unavailable(
                FeatureStatus::Unsupported,
                "rank-2 pair projection requires n>=2".into(),
            );
        }
        let Some(state_count) = pair_state_count(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion state count overflow".into(),
            );
        };
        if state_count > self.cache.config.max_pair_states {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "pair diffusion resource_limit: {state_count} states exceeds {}",
                    self.cache.config.max_pair_states
                ),
            );
        }
        let Some(profile) = self.cache.get_pair(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion checked allocation/iteration budget exceeded".into(),
            );
        };
        let Some(pair_count) = n.checked_mul(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output size overflow".into(),
            );
        };
        let Some(value_count) = pair_count.checked_mul(self.spec.channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output allocation failed".into(),
            );
        }
        values.resize(value_count, 0.0f32);

        for first in 0..n {
            for second in 0..n {
                if first == second {
                    continue;
                }
                let histogram = pair_histogram(state, first, second);
                let Some(class) = profile.class_index(&histogram) else {
                    return unavailable(
                        FeatureStatus::Inapplicable,
                        "matrix column pair is outside the rank-2 diffusion chain".into(),
                    );
                };
                let base = (first * n + second) * self.spec.channels;
                for time_index in 0..4 {
                    let raw = profile.raw(time_index, class);
                    let stationary = profile.stationary[class];
                    let exact_zero = profile.exact_zero[time_index * profile.states.len() + class];
                    let underflow =
                        (raw == 0.0 && !exact_zero) || (raw > 0.0 && (raw as f32) == 0.0);
                    let offset = base + time_index * 7;
                    values[offset] = raw as f32;
                    values[offset + 1] = profile.log_class_size[class] as f32;
                    values[offset + 2] = stationary as f32;
                    values[offset + 3] = (raw.max(self.cache.config.floor).ln()
                        - stationary.max(self.cache.config.floor).ln())
                        as f32;
                    values[offset + 4] = f32::from(exact_zero);
                    values[offset + 5] = f32::from(underflow);
                    values[offset + 6] = f32::from(raw < self.cache.config.floor);
                }
            }
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, n, self.spec.channels],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

fn unavailable(status: FeatureStatus, reason: String) -> FeatureComputation {
    FeatureComputation::Unavailable { status, reason }
}

fn row_weights(state: &MatrixState) -> Vec<usize> {
    let words = state.words_per_row() as usize;
    state
        .packed_rows()
        .chunks_exact(words)
        .map(|row| row.iter().map(|word| word.count_ones() as usize).sum())
        .collect()
}

fn column_weights(state: &MatrixState) -> Vec<usize> {
    let n = state.dimension() as usize;
    (0..n)
        .map(|column| {
            (0..n)
                .map(|row| matrix_bit(state, row, column) as usize)
                .sum()
        })
        .collect()
}

fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> bool {
    let words_per_row = state.words_per_row() as usize;
    let word = state.packed_rows()[row * words_per_row + column / 64];
    ((word >> (column % 64)) & 1) != 0
}

fn pair_histogram(state: &MatrixState, first: usize, second: usize) -> [usize; 4] {
    let n = state.dimension() as usize;
    let mut histogram = [0usize; 4];
    for row in 0..n {
        let pattern = (matrix_bit(state, row, first) as usize)
            | ((matrix_bit(state, row, second) as usize) << 1);
        histogram[pattern] += 1;
    }
    histogram
}
