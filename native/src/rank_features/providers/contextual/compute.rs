//! Evaluate contextual rank providers on a matrix state.

mod output;
mod rank_increment;
mod restricted_intersection;
mod selected_residual;

use super::{ContextualRankContext, ContextualRankProducer};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::packet::FeatureStatus;
use crate::rank_features::providers::helpers::unavailable;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

impl FeatureProducer for ContextualRankProducer {
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
                "restricted_rank_probes_require_augmentation_only_symmetry",
            );
        }
        match &self.context {
            ContextualRankContext::RestrictedIntersection { probes, transpose } => {
                restricted_intersection::compute(state, probes, *transpose)
            }
            ContextualRankContext::RankIncrement(probes) => rank_increment::compute(state, probes),
            ContextualRankContext::SelectedResidual { alphabet, gates } => {
                selected_residual::compute(state, alphabet, gates)
            }
        }
    }
}
