use super::output::{append_incidence, computed_group, group_buffers};
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::rank_features::contexts::RankIncrementProbe;
use crate::rank_features::providers::helpers::unavailable;
use crate::rank_features::rank_math::projected_rank;
use crate::state::MatrixState;

pub(super) fn compute(state: &MatrixState, probes: &[RankIncrementProbe]) -> FeatureComputation {
    let relevant = probes
        .iter()
        .filter(|probe| probe.n == state.dimension())
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return unavailable(
            FeatureStatus::Unsupported,
            "no_rank-increment_probe_for_matrix_width",
        );
    }
    let Some((mut values, mut incidence)) = group_buffers(state, relevant.len()) else {
        return unavailable(FeatureStatus::ResourceLimit, "rank-probe output too large");
    };
    for probe in &relevant {
        let baseline = match projected_rank(
            state,
            &probe.span_rows,
            &probe.coordinates,
            false,
            false,
            None,
        ) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        let augmented = match projected_rank(
            state,
            &probe.span_rows,
            &probe.coordinates,
            false,
            false,
            Some(&probe.candidate_support),
        ) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        let Some(increment) = augmented.checked_sub(baseline) else {
            return unavailable(
                FeatureStatus::Unsupported,
                "candidate reduced restricted rank",
            );
        };
        if increment > (1 << 24) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "rank_increment_exceeds_exact_f32_integer_range",
            );
        }
        values.push(increment as f32);
        append_incidence(
            state.dimension(),
            [
                probe.span_rows.as_slice(),
                probe.coordinates.as_slice(),
                probe.candidate_support.as_slice(),
            ],
            &mut incidence,
        );
    }
    computed_group(relevant.len(), values, incidence)
}
