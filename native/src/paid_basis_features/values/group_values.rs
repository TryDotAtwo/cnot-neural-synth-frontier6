use super::super::{PaidBasisBlockContext, WIRE_ROLES};
use super::exact_count;
use super::super::helpers::resource;
use crate::features::FeatureValue;
use crate::packet::FeatureStatus;
use crate::paid_basis::{BasisSide, PaidShearCandidate, BoundaryCancellation};
use std::result::Result;

pub(super) fn assemble_group_values(
    contexts: &[&PaidBasisBlockContext],
    candidates: &[PaidShearCandidate],
    n: usize,
) -> Result<
    (FeatureValue, FeatureValue, FeatureValue, FeatureValue, FeatureValue),
    (FeatureStatus, String),
> {
    let group_count = candidates.len();
    let incidence_count = group_count
        .checked_mul(n)
        .and_then(|value| value.checked_mul(WIRE_ROLES.len()))
        .ok_or_else(|| resource("paid_basis_incidence_size_overflow"))?;
    let mut incidence = Vec::new();
    incidence
        .try_reserve_exact(incidence_count)
        .map_err(|_| resource("paid_basis_incidence_allocation_failed"))?;
    for context in contexts {
        for role in [&context.source_wires, &context.target_wires] {
            let mut membership = vec![0.0f32; n];
            for wire in role {
                membership[*wire] = 1.0;
            }
            incidence.extend(membership);
        }
    }
    let token_valid = vec![1u8; group_count];

    let mut d09 = Vec::with_capacity(group_count);
    let mut d10 = Vec::with_capacity(group_count);
    let mut d11 = Vec::with_capacity(group_count * 2);
    let mut d12 = Vec::with_capacity(group_count * 2);
    let mut d13 = Vec::with_capacity(group_count * 12);
    for candidate in candidates {
        d09.push(exact_count(candidate.source_preparation.len())?);
        d10.push(exact_count(candidate.target_preparation.len())?);
        d11.extend([
            exact_count(candidate.core.len())?,
            exact_count(candidate.restoration.len())?,
        ]);
        d12.extend([
            exact_count(candidate.paid_cost)?,
            exact_count(candidate.direct_cost.saturating_sub(candidate.paid_cost))?,
        ]);

        let source_basis_changes = changed_from_identity(&candidate.source_basis)?;
        let target_basis_changes = changed_from_identity(&candidate.target_basis)?;
        let source_moves = candidate
            .accepted_moves
            .iter()
            .filter(|movement| movement.side == BasisSide::Source)
            .count();
        let target_moves = candidate
            .accepted_moves
            .iter()
            .filter(|movement| movement.side == BasisSide::Target)
            .count();
        let swaps = physical_swap_count(&candidate.certificate.gates);
        let unshortened = candidate
            .preparation
            .len()
            .checked_mul(2)
            .and_then(|value| value.checked_add(candidate.core.len()))
            .ok_or_else(|| resource("paid_basis_unshortened_cost_overflow"))?;
        let internal_cancellations = sum_cancellations(&candidate.boundary_cancellations)?;
        d13.extend([
            exact_count(candidate.source_preparation.len())?,
            exact_count(source_basis_changes)?,
            exact_count(source_moves)?,
            exact_count(candidate.target_preparation.len())?,
            exact_count(target_basis_changes)?,
            exact_count(target_moves)?,
            exact_count(swaps)?,
            exact_count(
                swaps
                    .checked_mul(3)
                    .ok_or_else(|| resource("paid_basis_swap_price_overflow"))?,
            )?,
            exact_count(candidate.direct_cost)?,
            exact_count(unshortened)?,
            exact_count(internal_cancellations)?,
            exact_count(candidate.paid_cost)?,
        ]);
    }

    let d09 = group_value(1, group_count, d09, token_valid.clone(), incidence.clone());
    let d10 = group_value(1, group_count, d10, token_valid.clone(), incidence.clone());
    let d11 = group_value(2, group_count, d11, token_valid.clone(), incidence.clone());
    let d12 = group_value(2, group_count, d12, token_valid.clone(), incidence.clone());
    let d13 = group_value(12, group_count, d13, token_valid, incidence);

    Ok((d09, d10, d11, d12, d13))
}

fn group_value(
    channels: usize,
    groups: usize,
    values: Vec<f32>,
    token_valid: Vec<u8>,
    incidence: Vec<f32>,
) -> FeatureValue {
    FeatureValue {
        shape: vec![groups, channels],
        values,
        token_valid,
        incidence,
    }
}

fn changed_from_identity(matrix: &[Vec<u8>]) -> Result<usize, (FeatureStatus, String)> {
    if matrix.iter().any(|row| row.len() != matrix.len()) {
        return Err(super::super::helpers::unsupported(
            "paid_basis_matrix_not_square",
        ));
    }
    Ok(matrix
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.iter()
                .enumerate()
                .filter(|(column, value)| **value != u8::from(row_index == *column))
                .count()
        })
        .sum())
}

fn physical_swap_count(gates: &[crate::state::CnotGate]) -> usize {
    let mut count = 0usize;
    let mut index = 0usize;
    while index + 2 < gates.len() {
        let first = gates[index];
        let middle = gates[index + 1];
        let last = gates[index + 2];
        let is_swap =
            first == last && first.control == middle.target && first.target == middle.control;
        if is_swap {
            count += 1;
            index += 3;
        } else {
            index += 1;
        }
    }
    count
}

fn sum_cancellations(
    cancellations: &[BoundaryCancellation],
) -> Result<usize, (FeatureStatus, String)> {
    cancellations.iter().try_fold(0usize, |sum, cancellation| {
        sum.checked_add(cancellation.canceled_pairs)
            .ok_or_else(|| resource("paid_basis_boundary_cancellation_overflow"))
    })
}
