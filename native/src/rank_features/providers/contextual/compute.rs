//! Evaluate contextual rank providers on a matrix state.
use super::{ContextualRankContext, ContextualRankProducer};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureValue};
use crate::packet::FeatureStatus;
use crate::rank_features::providers::helpers::unavailable;
use crate::rank_features::rank_math::projected_rank;
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
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "rank-probe output too large",
                    );
                };
                for probe in &relevant {
                    let left = match projected_rank(
                        state,
                        &probe.left,
                        &probe.coordinates,
                        false,
                        *transpose,
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
                        *transpose,
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
                        *transpose,
                        None,
                    ) {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    let Some(intersection) = left
                        .checked_add(right)
                        .and_then(|sum| sum.checked_sub(joint))
                    else {
                        return unavailable(
                            FeatureStatus::Unsupported,
                            "invalid subspace rank identity",
                        );
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
            ContextualRankContext::RankIncrement(probes) => {
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
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "rank-probe output too large",
                    );
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
            ContextualRankContext::SelectedResidual { alphabet, gates } => {
                if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id()
                {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "selected_residual_alphabet_differs_from_matrix_state",
                    );
                }
                let Some((mut values, mut incidence)) =
                    group_buffers_with_roles(state, gates.len(), 2)
                else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "rank-probe output too large",
                    );
                };
                let Some(wires) = all_wires(state.dimension()) else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "wire-index allocation failed",
                    );
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
                    let rank = match projected_rank(&transformed, &wires, &wires, true, false, None)
                    {
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
        }
    }
}

fn group_buffers(state: &MatrixState, groups: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    group_buffers_with_roles(state, groups, 3)
}

fn group_buffers_with_roles(
    state: &MatrixState,
    groups: usize,
    role_count: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let incidence_len = groups
        .checked_mul(state.dimension() as usize)?
        .checked_mul(role_count)?;
    let mut values = Vec::new();
    values.try_reserve_exact(groups).ok()?;
    let mut incidence = Vec::new();
    incidence.try_reserve_exact(incidence_len).ok()?;
    Some((values, incidence))
}

fn append_incidence(n: u32, roles: [&[u32]; 3], output: &mut Vec<f32>) {
    for wire in 0..n {
        for role in roles {
            output.push(f32::from(role.binary_search(&wire).is_ok()));
        }
    }
}

fn all_wires(n: u32) -> Option<Vec<u32>> {
    let length = usize::try_from(n).ok()?;
    let mut wires = Vec::new();
    wires.try_reserve_exact(length).ok()?;
    wires.extend(0..n);
    Some(wires)
}

fn computed_group(groups: usize, values: Vec<f32>, incidence: Vec<f32>) -> FeatureComputation {
    FeatureComputation::Computed(FeatureValue {
        shape: vec![groups, 1],
        values,
        token_valid: vec![1; groups],
        incidence,
    })
}
