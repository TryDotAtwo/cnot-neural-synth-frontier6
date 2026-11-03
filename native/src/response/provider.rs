//! Feature-provider adapter exposing the exact I10 oracle through the registry.

use super::contract::{I10Config, I10OracleStatus};
use super::oracle::{evaluate_i10, unavailable};
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;

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
