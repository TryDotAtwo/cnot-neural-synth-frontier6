use super::output::{append_incidence, computed_group, group_buffers};
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::rank_features::contexts::RestrictedIntersection;
use crate::rank_features::providers::helpers::unavailable;
use crate::rank_features::rank_math::projected_rank;
use crate::state::MatrixState;

pub(super) fn compute(
    state: &MatrixState,
    probes: &[RestrictedIntersection],
    transpose: bool,
) -> FeatureComputation {
    let relevant = probes
        .iter()
        .filter(|probe| probe.n == state.dimension())
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return unavailable(
            FeatureStatus::Unsupported,
            "no_restricted_intersection_probe_for_matrix_width",
        );
    }
    let Some((mut values, mut incidence)) = group_buffers(state, relevant.len()) else {
        return unavailable(FeatureStatus::ResourceLimit, "rank-probe output too large");
    };
    for probe in &relevant {
        let left = match projected_rank(
            state,
            &probe.left,
            &probe.coordinates,
            false,
            transpose,
            None,
        ) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        let right = match projected_rank(
            state,
            &probe.right,
            &probe.coordinates,
            false,
            transpose,
            None,
        ) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        let Some(joint_len) = probe.left.len().checked_add(probe.right.len()) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "joint rank row count overflow",
            );
        };
        let mut joint_rows = Vec::new();
        if joint_rows.try_reserve_exact(joint_len).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "joint rank row allocation failed",
            );
        }
        joint_rows.extend_from_slice(&probe.left);
        joint_rows.extend_from_slice(&probe.right);
        let joint = match projected_rank(
            state,
            &joint_rows,
            &probe.coordinates,
            false,
            transpose,
            None,
        ) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        let Some(intersection) = left
            .checked_add(right)
            .and_then(|sum| sum.checked_sub(joint))
        else {
            return unavailable(FeatureStatus::Unsupported, "invalid subspace rank identity");
        };
        if intersection > (1 << 24) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "intersection_dimension_exceeds_exact_f32_integer_range",
            );
        }
        values.push(intersection as f32);
        append_incidence(
            state.dimension(),
            [
                probe.left.as_slice(),
                probe.right.as_slice(),
                probe.coordinates.as_slice(),
            ],
            &mut incidence,
        );
    }
    computed_group(relevant.len(), values, incidence)
}
