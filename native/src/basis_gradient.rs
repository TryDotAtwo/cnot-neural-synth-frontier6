//! Exact zero-template partial-shear cost and gradient descriptors.
//!
//! These are augmentation-only because the source construction fixes the first
//! half of the physical wires as its source block. They describe the selected
//! partial elimination; they are not global lower bounds or optimality claims.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureSpec};
use crate::shear::partial_shear;
use crate::state::MatrixState;

#[derive(Clone, Copy)]
enum BasisGradientKind {
    Cost,
    PairGradient,
    GradientStatistics,
}

struct BasisGradientProducer {
    spec: FeatureSpec,
    kind: BasisGradientKind,
}

impl FeatureProducer for BasisGradientProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != self.symmetry_contract() {
            return unavailable("basis_gradient_requires_augmentation_only_symmetry");
        }
        let decomposition = match partial_shear(state) {
            Ok(Some(value)) => value,
            Ok(None) => return inapplicable("no_nonzero_partial_shear_block"),
            Err(_) => return unavailable("partial_shear_decomposition_failed"),
        };

        let width = decomposition.source_wires.len();
        if width < 2 || decomposition.block.len() != decomposition.target_wires.len() {
            return unavailable("partial_shear_block_has_invalid_shape");
        }
        if decomposition.block.iter().any(|row| row.len() != width) {
            return unavailable("partial_shear_block_has_invalid_shape");
        }

        let n = state.dimension() as usize;
        let mut column_weights = Vec::new();
        if column_weights.try_reserve_exact(width).is_err() {
            return resource_limit("partial_shear_column_weight_allocation_failed");
        }
        for column in 0..width {
            column_weights.push(
                decomposition
                    .block
                    .iter()
                    .map(|row| i64::from(row[column]))
                    .sum::<i64>(),
            );
        }

        let kind = match self.kind {
            BasisGradientKind::Cost => {
                let cost = decomposition
                    .block
                    .iter()
                    .flatten()
                    .map(|&bit| i64::from(bit))
                    .try_fold(0i64, i64::checked_add);
                let Some(cost) = cost else {
                    return resource_limit("partial_shear_cost_overflow");
                };
                FeatureValue {
                    shape: vec![1],
                    values: vec![cost as f32],
                    token_valid: Vec::new(),
                    incidence: Vec::new(),
                }
            }
            BasisGradientKind::PairGradient => {
                let Some(value_count) = n.checked_mul(n).and_then(|value| value.checked_mul(2))
                else {
                    return resource_limit("partial_shear_pair_output_size_overflow");
                };
                let mut values = Vec::new();
                if values.try_reserve_exact(value_count).is_err() {
                    return resource_limit("partial_shear_pair_output_allocation_failed");
                }
                values.resize(value_count, 0.0f32);
                for source in 0..width {
                    for target in 0..width {
                        if source == target {
                            continue;
                        }
                        let value = gradient(&decomposition.block, &column_weights, source, target);
                        let offset = (decomposition.source_wires[source] * n
                            + decomposition.source_wires[target])
                            * 2;
                        values[offset] = value as f32;
                        values[offset + 1] = 1.0;
                    }
                }
                FeatureValue {
                    shape: vec![n, n, 2],
                    values,
                    token_valid: Vec::new(),
                    incidence: Vec::new(),
                }
            }
            BasisGradientKind::GradientStatistics => {
                let Some(count) = width.checked_mul(width.saturating_sub(1)) else {
                    return resource_limit("partial_shear_gradient_count_overflow");
                };
                if count == 0 {
                    return unavailable("partial_shear_has_no_ordered_source_pairs");
                }
                let mut minimum = i64::MAX;
                let mut maximum = i64::MIN;
                let mut total = 0i64;
                let mut negative = 0usize;
                for source in 0..width {
                    for target in 0..width {
                        if source == target {
                            continue;
                        }
                        let value = gradient(&decomposition.block, &column_weights, source, target);
                        minimum = minimum.min(value);
                        maximum = maximum.max(value);
                        let Some(next_total) = total.checked_add(value) else {
                            return resource_limit("partial_shear_gradient_sum_overflow");
                        };
                        total = next_total;
                        negative += usize::from(value < 0);
                    }
                }
                FeatureValue {
                    shape: vec![4],
                    values: vec![
                        minimum as f32,
                        (total as f64 / count as f64) as f32,
                        maximum as f32,
                        negative as f32 / count as f32,
                    ],
                    token_valid: Vec::new(),
                    incidence: Vec::new(),
                }
            }
        };
        FeatureComputation::Computed(kind)
    }
}

fn unavailable(reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status: FeatureStatus::Unsupported,
        reason: reason.into(),
    }
}

fn inapplicable(reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status: FeatureStatus::Inapplicable,
        reason: reason.into(),
    }
}

fn resource_limit(reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status: FeatureStatus::ResourceLimit,
        reason: reason.into(),
    }
}

fn gradient(block: &[Vec<u8>], column_weights: &[i64], source: usize, target: usize) -> i64 {
    let overlap = block
        .iter()
        .map(|row| i64::from(row[source]) * i64::from(row[target]))
        .sum::<i64>();
    2 + column_weights[target] - 2 * overlap
}

/// Registers E01/E04/E05 for the source-defined zero-template partial shear.
pub fn register_basis_gradient(registry: &mut FeatureRegistry) -> Result<(), FeatureBuildError> {
    let ids = [
        FeatureId::new(b'E', 1).unwrap(),
        FeatureId::new(b'E', 4).unwrap(),
        FeatureId::new(b'E', 5).unwrap(),
    ];
    for id in ids {
        if registry.specs()[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let registrations = [
        (
            ids[0],
            BasisGradientKind::Cost,
            "partial_J17_zero_v1",
            "exact_integer_count",
        ),
        (
            ids[1],
            BasisGradientKind::PairGradient,
            "partial_J17_zero_gradient_v1",
            "float32_rounded_from_exact_counts",
        ),
        (
            ids[2],
            BasisGradientKind::GradientStatistics,
            "partial_J17_gradient_stats_v1",
            "float32_rounded_from_exact_counts",
        ),
    ];
    for (id, kind, producer_version, exactness) in registrations {
        let mut spec: FeatureSpec = feature_spec(id);
        spec.scope = "selected_partial_elimination;augmentation_only".into();
        spec.producer_version = producer_version.into();
        spec.dependencies = vec!["M".into(), "deterministic_partial_shear".into()];
        spec.exactness = exactness.into();
        spec.cost_tier = "T1".into();
        spec.wire_roles.clear();
        registry.register(Box::new(BasisGradientProducer { spec, kind }))?;
    }
    Ok(())
}
