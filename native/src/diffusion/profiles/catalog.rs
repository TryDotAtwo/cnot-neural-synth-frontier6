//! Cached finite-state profile representations.

mod column;
mod pair;

pub(in crate::diffusion) use column::ColumnProfile;
pub(in crate::diffusion) use pair::{pair_state_count, PairProfile};
