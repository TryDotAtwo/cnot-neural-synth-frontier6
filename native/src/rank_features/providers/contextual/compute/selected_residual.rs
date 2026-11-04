use super::output::{all_wires, computed_group, group_buffers_with_roles};
use crate::alphabet::CnotAlphabet;
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::rank_features::providers::helpers::unavailable;
use crate::rank_features::rank_math::projected_rank;
use crate::state::{CnotGate, MatrixState};

pub(super) fn compute(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    gates: &[CnotGate],
) -> FeatureComputation {
    if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
        return unavailable(
            FeatureStatus::Unsupported,
            "selected_residual_alphabet_differs_from_matrix_state",
        );
    }
    let Some((mut values, mut incidence)) = group_buffers_with_roles(state, gates.len(), 2) else {
        return unavailable(FeatureStatus::ResourceLimit, "rank-probe output too large");
    };
    let Some(wires) = all_wires(state.dimension()) else {
        return unavailable(FeatureStatus::ResourceLimit, "wire-index allocation failed");
    };
    for gate in gates {
        if alphabet.cost(*gate).is_none() {
            return unavailable(
                FeatureStatus::Unsupported,
                "selected_gate_missing_from_physical_alphabet",
            );
        }
        let mut transformed = state.clone();
        if transformed.apply_cnot(*gate).is_err() {
            return unavailable(
                FeatureStatus::Unsupported,
                "selected_gate_invalid_for_matrix_width",
            );
        }
        let rank = match projected_rank(&transformed, &wires, &wires, true, false, None) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        if rank > (1 << 24) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "selected_residual_rank_exceeds_exact_f32_integer_range",
            );
        }
        values.push(rank as f32);
        for wire in 0..state.dimension() {
            incidence.push(f32::from(gate.control == wire));
            incidence.push(f32::from(gate.target == wire));
        }
    }
    computed_group(gates.len(), values, incidence)
}
