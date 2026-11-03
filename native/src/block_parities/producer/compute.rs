mod group;
mod incidence;
mod node_frequency;
mod numeric;

use crate::block_parities::producer::patterns::PreparedBlock;
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::state::MatrixState;

pub(super) fn compute_group_feature(
    state: &MatrixState,
    ordinal: u8,
    channels: usize,
    wire_roles: &[String],
    prepared: &[PreparedBlock<'_>],
    max_pairs: usize,
) -> FeatureComputation {
    group::compute_group_feature(state, ordinal, channels, wire_roles, prepared, max_pairs)
}

pub(super) fn compute_node_frequencies(
    state: &MatrixState,
    prepared: &PreparedBlock<'_>,
) -> FeatureComputation {
    node_frequency::compute_node_frequencies(state, prepared)
}

pub(super) fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
