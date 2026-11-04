//! Budgeted exact endpoint word-series computation for I10.

use super::contract::{I10Config, I10OracleResult, I10OracleStatus};
use crate::alphabet::CnotAlphabet;
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::state::MatrixState;
use std::collections::HashMap;

type ExactWordSeries = Result<(Vec<(usize, u128)>, usize), (String, usize)>;

/// Evaluates the source-defined exact positive endpoint word series for I10.
///
/// This exposes the same scope, transition count, status, and exact degree
/// coefficients used to form the four-channel feature. A budget failure
/// discards the partial series.
pub fn evaluate_i10(target: &MatrixState, config: I10Config) -> I10OracleResult {
    let n = target.dimension();
    let scope = format!("full_cnot_alphabet:n{n}");
    if config.max_n < 2 {
        return oracle_result(
            I10OracleStatus::Unsupported,
            Vec::new(),
            0,
            "configured_max_n_must_be_at_least_two",
            scope,
        );
    }
    if n < 2 {
        return oracle_result(
            I10OracleStatus::Unsupported,
            Vec::new(),
            0,
            "response_requires_width_at_least_two",
            scope,
        );
    }
    if n > config.max_n {
        return oracle_result(
            I10OracleStatus::Unsupported,
            Vec::new(),
            0,
            "width_exceeds_configured_max_n",
            scope,
        );
    }
    let alphabet = match CnotAlphabet::all_to_all_unit(n) {
        Ok(alphabet) => alphabet,
        Err(error) => {
            return oracle_result(
                I10OracleStatus::Unsupported,
                Vec::new(),
                0,
                &format!("invalid_full_cnot_alphabet:{error}"),
                scope,
            )
        }
    };
    if target.alphabet_id() != alphabet.id() {
        return oracle_result(
            I10OracleStatus::Unsupported,
            Vec::new(),
            0,
            "state_alphabet_is_not_the_full_unit_cnot_alphabet",
            scope,
        );
    }

    match exact_word_series(target, &alphabet, config) {
        Ok((positive_terms, evaluations)) => oracle_result(
            I10OracleStatus::Exact,
            positive_terms,
            evaluations,
            "complete_through_requested_degree",
            scope,
        ),
        Err((reason, evaluations)) => oracle_result(
            I10OracleStatus::ResourceLimit,
            Vec::new(),
            evaluations,
            &reason,
            scope,
        ),
    }
}

fn oracle_result(
    status: I10OracleStatus,
    positive_terms: Vec<(usize, u128)>,
    evaluations: usize,
    reason: &str,
    scope: String,
) -> I10OracleResult {
    I10OracleResult {
        status,
        positive_terms,
        evaluations,
        reason: reason.into(),
        scope,
    }
}

pub(super) fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn exact_word_series(
    target: &MatrixState,
    alphabet: &CnotAlphabet,
    config: I10Config,
) -> ExactWordSeries {
    let mut layer = HashMap::<Vec<u64>, u128>::new();
    let identity = MatrixState::identity(target.dimension(), target.alphabet_id())
        .map_err(|error| (format!("invalid_identity_state:{error}"), 0))?;
    layer.insert(identity.packed_rows().to_vec(), 1);
    let mut positive_terms = Vec::new();
    if target == &identity {
        positive_terms.push((0, 1));
    }
    let target_rows = target.packed_rows();
    let mut evaluations = 0usize;

    for degree in 1..=config.max_degree {
        if layer.len() > config.max_layer_states {
            return Err((
                "complete_next_layer_exceeds_declared_budget".into(),
                evaluations,
            ));
        }
        let required = layer
            .len()
            .checked_mul(alphabet.gates().len())
            .ok_or_else(|| ("transition_count_overflow".into(), evaluations))?;
        if required > config.max_transitions.saturating_sub(evaluations) {
            return Err((
                "complete_next_layer_exceeds_declared_budget".into(),
                evaluations,
            ));
        }

        let mut next_layer = HashMap::<Vec<u64>, u128>::new();
        for (rows, count) in &layer {
            for gate in alphabet.gates() {
                let control_start = gate.control as usize * target.words_per_row();
                let target_start = gate.target as usize * target.words_per_row();
                let control_row =
                    rows[control_start..control_start + target.words_per_row()].to_vec();
                let mut moved = rows.clone();
                for (offset, word) in control_row.into_iter().enumerate() {
                    moved[target_start + offset] ^= word;
                }
                let entry = next_layer.entry(moved).or_insert(0);
                *entry = entry
                    .checked_add(*count)
                    .ok_or_else(|| ("exact_coefficient_overflow_u128".into(), evaluations))?;
                evaluations += 1;
            }
        }
        if next_layer.len() > config.max_layer_states {
            return Err((
                "complete_next_layer_exceeds_declared_budget".into(),
                evaluations,
            ));
        }
        if let Some(multiplicity) = next_layer.get(target_rows).copied() {
            if multiplicity > 0 {
                positive_terms.push((degree, multiplicity));
            }
        }
        layer = next_layer;
    }

    Ok((positive_terms, evaluations))
}
