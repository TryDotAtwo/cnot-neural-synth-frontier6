//! Exact finite-horizon I10 endpoint response for the full ordered CNOT alphabet.
//!
//! A completed layer counts every physical gate word exactly. If the requested
//! layer exceeds a declared state/transition budget, or an exact coefficient
//! exceeds the native counter width, I10 is unavailable; partial counts are
//! never exposed as exact features.

use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct I10Config {
    pub max_n: u32,
    pub max_degree: usize,
    pub max_layer_states: usize,
    pub max_transitions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum I10OracleStatus {
    Exact,
    Unsupported,
    ResourceLimit,
}

/// Direct oracle result. `positive_terms` is empty on non-exact outcomes and
/// otherwise contains `(physical_degree, exact_word_count)` in degree order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct I10OracleResult {
    pub status: I10OracleStatus,
    pub positive_terms: Vec<(usize, u128)>,
    pub evaluations: usize,
    pub reason: String,
    pub scope: String,
}

/// Registers the source-compatible exact-leading I10 projection.
///
/// The feature reports the minimum positive physical CNOT degree through the
/// configured finite horizon, `log(1 + exact multiplicity)`, an exact-positive
/// flag, and zero interval width. A horizon miss is unavailable, not zero.
pub fn register_tiny_i10(
    registry: &mut FeatureRegistry,
    config: I10Config,
) -> Result<(), FeatureBuildError> {
    if config.max_n < 2 {
        return Err(FeatureBuildError(
            "I10 requires an explicit maximum width of at least two".into(),
        ));
    }
    let id = FeatureId::new(b'I', 10).expect("I10 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 4;
    spec.scope = "solver_assisted:full_cnot_alphabet:exact_leading_only".into();
    spec.producer_version = format!(
        "tiny_full_i10_solver_assisted_v2_u128_n{}_d{}_s{}_t{}",
        config.max_n, config.max_degree, config.max_layer_states, config.max_transitions
    );
    spec.dependencies = vec![
        "M".into(),
        "unit_full_alphabet".into(),
        "local_budget".into(),
    ];
    spec.exactness = "derived_from_exact_positive_coefficient".into();
    spec.cost_tier = "T3".into();
    spec.wire_roles.clear();
    spec.requires_solver_assisted = true;
    registry.register(Box::new(TinyI10Producer { spec, config }))
}

struct TinyI10Producer {
    spec: FeatureSpec,
    config: I10Config,
}

impl FeatureProducer for TinyI10Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if !policy.solver_assisted {
            return unavailable(
                FeatureStatus::Unsupported,
                "solver_assisted_policy_required",
            );
        }
        let result = evaluate_i10(state, self.config);
        match result.status {
            I10OracleStatus::Exact => match result.positive_terms.first().copied() {
                Some((degree, multiplicity)) => FeatureComputation::Computed(FeatureValue {
                    shape: vec![4],
                    values: vec![
                        degree as f32,
                        (multiplicity as f64).ln_1p() as f32,
                        1.0,
                        0.0,
                    ],
                    token_valid: Vec::new(),
                    incidence: Vec::new(),
                }),
                None => unavailable(
                    FeatureStatus::Unsupported,
                    "no_positive_term_within_horizon",
                ),
            },
            I10OracleStatus::Unsupported => unavailable(FeatureStatus::Unsupported, &result.reason),
            I10OracleStatus::ResourceLimit => {
                unavailable(FeatureStatus::ResourceLimit, &result.reason)
            }
        }
    }
}

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

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn exact_word_series(
    target: &MatrixState,
    alphabet: &CnotAlphabet,
    config: I10Config,
) -> Result<(Vec<(usize, u128)>, usize), (String, usize)> {
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
