//! Context-bound rank producer types and registrations.
mod compute;
mod registration;

use crate::alphabet::CnotAlphabet;
use crate::rank_features::contexts::{RankIncrementProbe, RestrictedIntersection};
use crate::schema::FeatureSpec;
use crate::state::CnotGate;

pub use registration::{
    register_rank_increment_probes, register_restricted_intersections,
    register_selected_residual_ranks,
};

#[derive(Clone)]
pub(super) enum ContextualRankContext {
    RestrictedIntersection {
        probes: Vec<RestrictedIntersection>,
        transpose: bool,
    },
    RankIncrement(Vec<RankIncrementProbe>),
    SelectedResidual {
        alphabet: CnotAlphabet,
        gates: Vec<CnotGate>,
    },
}

pub(super) struct ContextualRankProducer {
    pub(super) spec: FeatureSpec,
    pub(super) context: ContextualRankContext,
}
