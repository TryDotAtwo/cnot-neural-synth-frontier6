//! Exact cost, pair-gradient, and aggregate-gradient calculations.

use super::{inapplicable, resource_limit, unavailable, BasisGradientKind};
use crate::features::{FeatureComputation, FeatureValue};
use crate::shear::partial_shear;
use crate::state::MatrixState;

pub(super) fn compute(kind: BasisGradientKind, state: &MatrixState) -> FeatureComputation {
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

    let kind_value = match kind {
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
            let Some(value_count) = n.checked_mul(n).and_then(|value| value.checked_mul(2)) else {
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
    FeatureComputation::Computed(kind_value)
}

fn gradient(block: &[Vec<u8>], column_weights: &[i64], source: usize, target: usize) -> i64 {
    let overlap = block
        .iter()
        .map(|row| i64::from(row[source]) * i64::from(row[target]))
        .sum::<i64>();
    2 + column_weights[target] - 2 * overlap
}
