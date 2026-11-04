//! Frozen-data wire schema and semantic decoding.

mod model;
mod panel;
mod state;

pub(super) use model::Body;
#[cfg(test)]
pub(super) use model::State;
pub(super) use panel::panel;
#[cfg(test)]
pub(super) use state::decode_state;
pub(super) use state::key_set;
