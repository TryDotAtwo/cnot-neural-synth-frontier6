//! Pack exact paid-basis results into the D09-D14 feature layouts.
use super::helpers::{resource, unsupported};
use super::{PaidBasisBlockContext, WIRE_ROLES};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::features::FeatureValue;
use crate::packet::FeatureStatus;
use crate::paid_basis::{
    BasisSide, BoundaryCancellation, PaidShearCandidate, cancel_adjacent_block_gates,
    endpoint_sha256,
};
use crate::state::{CnotGate, MatrixState};
use std::collections::HashMap;

#[derive(Clone)]
pub(super) struct PaidBasisFeatureValues {
    pub(super) d09: FeatureValue,
    pub(super) d10: FeatureValue,
    pub(super) d11: FeatureValue,
    pub(super) d12: FeatureValue,
    pub(super) d13: FeatureValue,
    pub(super) d14: FeatureValue,
    pub(super) d14_provenance: Vec<String>,
}

pub(super) fn assemble_feature_values(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    contexts: &[&PaidBasisBlockContext],
    candidates: &[PaidShearCandidate],
) -> Result<PaidBasisFeatureValues, (FeatureStatus, String)> {
    let n = state.dimension() as usize;
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

    let word_blocks = candidates
        .iter()
        .map(|candidate| candidate.certificate.gates.clone())
        .collect::<Vec<_>>();
    let compatibility = adjacent_compatibility_counts(&word_blocks);
    let reduced = cancel_adjacent_block_gates(&word_blocks);
    let start = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|error| unsupported(error.to_string()))?;
    let mut endpoint = start.clone();
    for gate in &reduced.word {
        endpoint
            .apply_cnot(*gate)
            .map_err(|error| unsupported(error.to_string()))?;
    }
    let combined_certificate = WordCertificate {
        start,
        end: endpoint.clone(),
        physical_cost: u64::try_from(reduced.word.len())
            .map_err(|_| resource("paid_basis_combined_word_length_exceeds_u64"))?,
        gates: reduced.word.clone(),
        producer: "verified_paid_basis_context_sequence_v1".into(),
        lineage_id: "explicit_paid_basis_contexts".into(),
    };
    combined_certificate
        .verify(alphabet)
        .map_err(|error| unsupported(error.to_string()))?;
    let endpoint_digest = endpoint_sha256(&endpoint);

    let pair_count = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| resource("paid_basis_d14_size_overflow"))?;
    let mut pair_values = Vec::new();
    pair_values
        .try_reserve_exact(pair_count)
        .map_err(|_| resource("paid_basis_d14_allocation_failed"))?;
    pair_values.resize(pair_count, 0.0f32);
    let mut gate_counts = HashMap::<(u32, u32), usize>::new();
    for gate in &reduced.word {
        let count = gate_counts.entry((gate.control, gate.target)).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| resource("paid_basis_d14_gate_count_overflow"))?;
    }
    let compatibility_counts = compatibility
        .into_iter()
        .map(|(gate, count)| (gate, count))
        .collect::<HashMap<_, _>>();
    let cancellation_counts = reduced
        .boundary_cancellations
        .iter()
        .map(|cancellation| {
            (
                (cancellation.control, cancellation.target),
                cancellation.canceled_pairs,
            )
        })
        .collect::<HashMap<_, _>>();
    for control in 0..n {
        for target in 0..n {
            let offset = (control * n + target) * 4;
            let endpoint_bit = matrix_bit(&endpoint, target, control);
            pair_values[offset] = f32::from(endpoint_bit);
            pair_values[offset + 1] = exact_count(
                gate_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
            pair_values[offset + 2] = exact_count(
                compatibility_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
            pair_values[offset + 3] = exact_count(
                cancellation_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
        }
    }
    let mut provenance = vec![format!("endpoint_sha256={endpoint_digest}")];
    for (index, candidate) in candidates.iter().enumerate() {
        provenance.push(format!(
            "block_{index}_endpoint_sha256={}",
            candidate.endpoint_sha256
        ));
    }
    let d14 = FeatureValue {
        shape: vec![n, n, 4],
        values: pair_values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    };

    Ok(PaidBasisFeatureValues {
        d09,
        d10,
        d11,
        d12,
        d13,
        d14,
        d14_provenance: provenance,
    })
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
        return Err(unsupported("paid_basis_matrix_not_square"));
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

fn physical_swap_count(gates: &[CnotGate]) -> usize {
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

fn adjacent_compatibility_counts(blocks: &[Vec<CnotGate>]) -> HashMap<(u32, u32), usize> {
    let mut counts = HashMap::new();
    for adjacent in blocks.windows(2) {
        for (left, right) in adjacent[0].iter().rev().zip(&adjacent[1]) {
            if left != right {
                break;
            }
            let count = counts.entry((left.control, left.target)).or_insert(0usize);
            *count += 1;
        }
    }
    counts
}

pub(super) fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> u8 {
    let words = state
        .row_words(row as u32)
        .expect("row index is bounded by the matrix dimension");
    ((words[column / 64] >> (column % 64)) & 1) as u8
}

fn exact_count(value: usize) -> Result<f32, (FeatureStatus, String)> {
    if value > (1usize << 24) {
        return Err(resource("paid_basis_integer_exceeds_exact_f32_range"));
    }
    Ok(value as f32)
}

fn sum_cancellations(
    cancellations: &[BoundaryCancellation],
) -> Result<usize, (FeatureStatus, String)> {
    cancellations.iter().try_fold(0usize, |sum, cancellation| {
        sum.checked_add(cancellation.canceled_pairs)
            .ok_or_else(|| resource("paid_basis_boundary_cancellation_overflow"))
    })
}
