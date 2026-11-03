//! Exact GF(2) rank features for explicit cuts, candidates, and physical actions.
//!
//! Contextual values are augmentation-only: none is invariant under arbitrary
//! wire relabeling. A cut, candidate, or physical action is never guessed.

mod contexts;
mod providers;
mod rank_math;

pub use contexts::{LabeledCut, LabeledCutError, RankIncrementProbe, RestrictedIntersection};
pub use providers::{
    register_labeled_cuts, register_partial_shear_rank, register_rank_increment_probes,
    register_restricted_intersections, register_selected_residual_ranks,
};
