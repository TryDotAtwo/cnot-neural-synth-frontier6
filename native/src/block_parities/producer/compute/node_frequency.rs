use super::super::patterns::PreparedBlock;
use super::numeric::exact_f32_integer;
use super::unavailable;
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
