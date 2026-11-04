//! Rank feature for the source-defined deterministic partial-shear block.
use super::helpers::{ensure_unconfigured, unavailable};
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::rank_features::rank_math::rank_binary_block;
use crate::schema::{feature_spec, FeatureId, FeatureSpec};
use crate::shear::partial_shear;
use crate::state::MatrixState;

struct PartialShearRankProducer {
    spec: FeatureSpec,
}

impl FeatureProducer for PartialShearRankProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "partial_shear_rank_requires_augmentation_only_symmetry",
            );
        }
        let decomposition = match partial_shear(state) {
            Ok(Some(decomposition)) => decomposition,
            Ok(None) => {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "no_nonzero_partial_shear_block",
                );
            }
            Err(_) => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "partial_shear_decomposition_failed",
                );
            }
        };
        let rank = match rank_binary_block(&decomposition.block) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        if rank > (1 << 24) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "partial_shear_rank_exceeds_exact_f32_integer_range",
            );
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![1],
            values: vec![rank as f32],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

/// Registers C05 for the source-defined deterministic partial-shear block.
pub fn register_partial_shear_rank(
    registry: &mut FeatureRegistry,
) -> Result<(), FeatureBuildError> {
    let id = FeatureId::new(b'C', 5).unwrap();
    ensure_unconfigured(registry, &[id])?;
    let mut spec = feature_spec(id);
    spec.scope = "selected_partial_elimination;augmentation_only".into();
    spec.producer_version = "selected_partial_shear_rank_v1".into();
    spec.dependencies = vec!["M".into(), "deterministic_partial_shear".into()];
    spec.exactness = "exact".into();
    spec.cost_tier = "T1".into();
    spec.wire_roles.clear();
    registry.register(Box::new(PartialShearRankProducer { spec }))
}
