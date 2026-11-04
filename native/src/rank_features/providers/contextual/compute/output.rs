use crate::features::{FeatureComputation, FeatureValue};
use crate::state::MatrixState;

pub(super) fn group_buffers(state: &MatrixState, groups: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    group_buffers_with_roles(state, groups, 3)
}

pub(super) fn group_buffers_with_roles(
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

pub(super) fn append_incidence(n: u32, roles: [&[u32]; 3], output: &mut Vec<f32>) {
    for wire in 0..n {
        for role in roles {
            output.push(f32::from(role.binary_search(&wire).is_ok()));
        }
    }
}

pub(super) fn all_wires(n: u32) -> Option<Vec<u32>> {
    let length = usize::try_from(n).ok()?;
    let mut wires = Vec::new();
    wires.try_reserve_exact(length).ok()?;
    wires.extend(0..n);
    Some(wires)
}

pub(super) fn computed_group(
    groups: usize,
    values: Vec<f32>,
    incidence: Vec<f32>,
) -> FeatureComputation {
    FeatureComputation::Computed(FeatureValue {
        shape: vec![groups, 1],
        values,
        token_valid: vec![1; groups],
        incidence,
    })
}
