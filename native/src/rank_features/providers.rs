//! Registrations for exact rank feature providers with explicit contexts.
mod contextual;
mod helpers;
mod labeled_cuts;
mod partial_shear;

pub use contextual::{
    register_rank_increment_probes, register_restricted_intersections,
    register_selected_residual_ranks,
};
pub use labeled_cuts::register_labeled_cuts;
pub use partial_shear::register_partial_shear_rank;
