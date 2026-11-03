use super::gf2::rank_patterns;
use super::patterns::{ParityPattern, PreparedBlock};
use crate::block_parities::BlockContext;
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use crate::state::MatrixState;
use std::collections::HashMap;

pub(super) fn compute_node_frequencies(
    state: &MatrixState,
    prepared: &PreparedBlock<'_>,
) -> FeatureComputation {
    let n = state.dimension() as usize;
    let Some(value_len) = n.checked_mul(2) else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "node block feature size overflow",
        );
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(value_len).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "node block feature allocation failed",
        );
    }
    values.resize(value_len, 0.0f32);
    for (target, code) in prepared.context.targets.iter().zip(&prepared.data.codes) {
        let Some(count) = prepared.data.counts.get(code).copied() else {
            return unavailable(
                FeatureStatus::Unsupported,
                "target parity pattern is unindexed",
            );
        };
        let Some(count) = exact_f32_integer(count) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "target frequency exceeds exact f32 range",
            );
        };
        values[*target as usize * 2] = count;
    }

    let mut column_counts = HashMap::new();
    if column_counts
        .try_reserve(prepared.data.columns.len())
        .is_err()
    {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "column-frequency index allocation failed",
        );
    }
    for code in &prepared.data.columns {
        let entry = column_counts.entry(code).or_insert(0usize);
        let Some(next) = entry.checked_add(1) else {
            return unavailable(FeatureStatus::ResourceLimit, "column frequency overflow");
        };
        *entry = next;
    }
    for (source, code) in prepared.context.sources.iter().zip(&prepared.data.columns) {
        let Some(count) = column_counts.get(code).copied() else {
            return unavailable(
                FeatureStatus::Unsupported,
                "source parity column is unindexed",
            );
        };
        let Some(count) = exact_f32_integer(count) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column frequency exceeds exact f32 range",
            );
        };
        values[*source as usize * 2 + 1] = count;
    }
    FeatureComputation::Computed(FeatureValue {
        shape: vec![n, 2],
        values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}

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
        match ordinal {
            1 | 2 | 4 => {
                for pattern in &block.patterns {
                    let count = block.data.counts[pattern];
                    let value = match ordinal {
                        1 => count,
                        2 => 1,
                        4 if !pattern.is_zero() => count,
                        4 => 0,
                        _ => unreachable!(),
                    };
                    let Some(value) = exact_f32_integer(value) else {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "pattern frequency exceeds exact f32 range",
                        );
                    };
                    values.push(value);
                    append_incidence(
                        block.context,
                        &block.data.codes,
                        Some(pattern),
                        IncidenceColumn::MatchingTargets(pattern),
                        &mut incidence,
                    );
                    token_valid.push(1);
                }
            }
            6 => {
                let nonzero = block
                    .patterns
                    .iter()
                    .filter(|pattern| !pattern.is_zero())
                    .collect::<Vec<_>>();
                for left_index in 0..nonzero.len() {
                    for right_index in (left_index + 1)..nonzero.len() {
                        let left = nonzero[left_index];
                        let right = nonzero[right_index];
                        let Some(value) = exact_f32_integer(left.intersection_count(right)) else {
                            return unavailable(
                                FeatureStatus::ResourceLimit,
                                "pattern intersection exceeds exact f32 range",
                            );
                        };
                        values.push(value);
                        append_incidence(
                            block.context,
                            &block.data.codes,
                            Some(left),
                            IncidenceColumn::SourcePattern(right),
                            &mut incidence,
                        );
                        token_valid.push(1);
                    }
                }
            }
            7 => {
                let nonzero = block
                    .patterns
                    .iter()
                    .filter(|pattern| !pattern.is_zero())
                    .collect::<Vec<_>>();
                let rank = match rank_patterns(&nonzero, block.context.sources.len()) {
                    Ok(rank) => rank,
                    Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                };
                let total = nonzero.iter().try_fold(0usize, |sum, pattern| {
                    sum.checked_add(block.data.counts[*pattern])
                });
                let Some(total) = total else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "nonzero pattern count overflow",
                    );
                };
                let max_frequency = nonzero
                    .iter()
                    .map(|pattern| block.data.counts[*pattern])
                    .max()
                    .unwrap_or(0);
                let scalars = [
                    rank,
                    nonzero.len(),
                    total,
                    total - nonzero.len(),
                    max_frequency,
                ];
                for scalar in scalars {
                    let Some(value) = exact_f32_integer(scalar) else {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "block rank summary exceeds exact f32 range",
                        );
                    };
                    values.push(value);
                }
                append_incidence(
                    block.context,
                    &block.data.codes,
                    None,
                    IncidenceColumn::AllTargets,
                    &mut incidence,
                );
                token_valid.push(1);
            }
            8 => {
                let parity_weight = block
                    .data
                    .codes
                    .iter()
                    .flat_map(|pattern| &pattern.0)
                    .map(|word| u128::from(word.count_ones()))
                    .sum::<u128>();
                let Some(value) = exact_f32_integer_u128(parity_weight) else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "block parity weight exceeds exact f32 range",
                    );
                };
                values.push(value);
                append_incidence(
                    block.context,
                    &block.data.codes,
                    None,
                    IncidenceColumn::AllTargets,
                    &mut incidence,
                );
                token_valid.push(1);
            }
            _ => unreachable!(),
        }
    }
    FeatureComputation::Computed(FeatureValue {
        shape: vec![group_count, channels],
        values,
        token_valid,
        incidence,
    })
}

enum IncidenceColumn<'a> {
    SourcePattern(&'a ParityPattern),
    MatchingTargets(&'a ParityPattern),
    AllTargets,
}

fn append_incidence(
    context: &BlockContext,
    codes: &[ParityPattern],
    role_one_pattern: Option<&ParityPattern>,
    role_two: IncidenceColumn<'_>,
    output: &mut Vec<f32>,
) {
    for wire in 0..context.n {
        let source_position = context.sources.binary_search(&wire).ok();
        output.push(f32::from(source_position.is_some()));
        let role_one = role_one_pattern
            .and_then(|pattern| source_position.map(|index| pattern.bit(index)))
            .unwrap_or_else(|| role_one_pattern.is_none() && source_position.is_some());
        output.push(f32::from(role_one));
        let role_two = match &role_two {
            IncidenceColumn::SourcePattern(pattern) => source_position
                .map(|index| pattern.bit(index))
                .unwrap_or(false),
            IncidenceColumn::MatchingTargets(pattern) => context
                .targets
                .binary_search(&wire)
                .ok()
                .and_then(|index| codes.get(index))
                .is_some_and(|code| code == *pattern),
            IncidenceColumn::AllTargets => context.targets.binary_search(&wire).is_ok(),
        };
        output.push(f32::from(role_two));
    }
}

fn exact_f32_integer(value: usize) -> Option<f32> {
    exact_f32_integer_u128(value as u128)
}

fn exact_f32_integer_u128(value: u128) -> Option<f32> {
    (value <= (1 << 24)).then_some(value as f32)
}

pub(super) fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
