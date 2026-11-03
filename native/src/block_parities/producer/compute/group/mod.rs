mod intersections;
mod patterns;
mod summaries;

use super::unavailable;
use crate::block_parities::producer::patterns::PreparedBlock;
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use crate::state::MatrixState;

pub(super) fn compute_group_feature(
    state: &MatrixState,
    ordinal: u8,
    channels: usize,
    wire_roles: &[String],
    prepared: &[PreparedBlock<'_>],
    max_pairs: usize,
) -> FeatureComputation {
    let mut group_count = 0usize;
    for block in prepared {
        let count = match ordinal {
            1 | 2 | 4 => block.patterns.len(),
            6 => {
                let nonzero = block
                    .patterns
                    .iter()
                    .filter(|pattern| !pattern.is_zero())
                    .count();
                match nonzero
                    .checked_mul(nonzero.saturating_sub(1))
                    .map(|pairs| pairs / 2)
                {
                    Some(pairs) if pairs <= max_pairs => pairs,
                    Some(_) => {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "nonzero_pattern_pair_budget_exceeded",
                        );
                    }
                    None => {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "pattern pair count overflow",
                        );
                    }
                }
            }
            7 | 8 => 1,
            _ => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "unsupported block parity feature",
                );
            }
        };
        let Some(next) = group_count.checked_add(count) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "block feature group count overflow",
            );
        };
        group_count = next;
    }
    let n = state.dimension() as usize;
    let Some(value_count) = group_count.checked_mul(channels) else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block feature value size overflow",
        );
    };
    let Some(incidence_count) = group_count
        .checked_mul(n)
        .and_then(|size| size.checked_mul(wire_roles.len()))
    else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block feature incidence size overflow",
        );
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(value_count).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block feature value allocation failed",
        );
    }
    let mut incidence = Vec::new();
    if incidence.try_reserve_exact(incidence_count).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block incidence allocation failed",
        );
    }
    let mut token_valid = Vec::new();
    if token_valid.try_reserve_exact(group_count).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block token allocation failed",
        );
    }

    for block in prepared {
        let result = match ordinal {
            1 | 2 | 4 => patterns::append(
                ordinal,
                block,
                &mut values,
                &mut incidence,
                &mut token_valid,
            ),
            6 => intersections::append(block, &mut values, &mut incidence, &mut token_valid),
            7 => {
                summaries::append_rank_summary(block, &mut values, &mut incidence, &mut token_valid)
            }
            8 => summaries::append_parity_weight(
                block,
                &mut values,
                &mut incidence,
                &mut token_valid,
            ),
            _ => unreachable!(),
        };
        if let Err(reason) = result {
            return unavailable(FeatureStatus::ResourceLimit, reason);
        }
    }
    FeatureComputation::Computed(FeatureValue {
        shape: vec![group_count, channels],
        values,
        token_valid,
        incidence,
    })
}
