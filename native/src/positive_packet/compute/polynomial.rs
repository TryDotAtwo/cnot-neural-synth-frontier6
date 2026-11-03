//! Exact optional-gate polynomial expansion and affine packet extraction.

use super::storage::{highest_set_bit, set_bit, words_for, zeroed_u64};
use super::ComputeFailure;
use crate::positive_packet::OptionalGatePacket;

#[derive(Clone, Debug)]
pub(super) struct PolynomialTerm {
    mask: Vec<u64>,
    matrix: Vec<u64>,
}

pub(super) fn affine_columns(
    context: &OptionalGatePacket,
    max_monomials: usize,
) -> Result<Vec<Vec<u64>>, ComputeFailure> {
    let n = usize::try_from(context.n).map_err(|_| ComputeFailure::ResourceLimit)?;
    let slots = context.gates.len();
    let words_per_row = n.checked_add(63).ok_or(ComputeFailure::ResourceLimit)? / 64;
    let matrix_words = n
        .checked_mul(words_per_row)
        .ok_or(ComputeFailure::ResourceLimit)?;
    let mask_words = words_for(slots).ok_or(ComputeFailure::ResourceLimit)?;

    // Expand (I + x_j E_j) in the exact declared physical gate order. A term
    // produced at slot j receives a fresh bit, so distinct histories cannot
    // collide in the polynomial map.
    let mut identity = zeroed_u64(matrix_words)?;
    for row in 0..n {
        identity[row * words_per_row + row / 64] |= 1u64 << (row % 64);
    }
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(1)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    terms.push(PolynomialTerm {
        mask: zeroed_u64(mask_words)?,
        matrix: identity,
    });

    for (slot, (control, target)) in context.gates.iter().copied().enumerate() {
        let control = usize::try_from(control).map_err(|_| ComputeFailure::ResourceLimit)?;
        let target = usize::try_from(target).map_err(|_| ComputeFailure::ResourceLimit)?;
        let original_len = terms.len();
        let remaining = max_monomials.saturating_sub(original_len);
        let mut extra = Vec::new();
        extra
            .try_reserve_exact(remaining.min(original_len))
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        for term in &terms[..original_len] {
            let source_start = control
                .checked_mul(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            let source_end = source_start
                .checked_add(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            let source_row = &term.matrix[source_start..source_end];
            if source_row.iter().all(|word| *word == 0) {
                continue;
            }
            if extra.len() >= remaining {
                return Err(ComputeFailure::ResourceLimit);
            }
            let mut mask = term.mask.clone();
            set_bit(&mut mask, slot);
            let mut matrix = zeroed_u64(matrix_words)?;
            let target_start = target
                .checked_mul(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            let target_end = target_start
                .checked_add(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            matrix[target_start..target_end].copy_from_slice(source_row);
            extra.push(PolynomialTerm { mask, matrix });
        }
        terms
            .try_reserve(extra.len())
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        terms.extend(extra);
    }

    let flat_bits = n.checked_mul(n).ok_or(ComputeFailure::ResourceLimit)?;
    let flat_words = words_for(flat_bits).ok_or(ComputeFailure::ResourceLimit)?;
    let mut singleton_terms = Vec::new();
    singleton_terms
        .try_reserve_exact(slots)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    singleton_terms.resize(slots, None);
    let mut has_constant = false;
    for term in &terms {
        let degree = term
            .mask
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if degree > 1 {
            return Err(ComputeFailure::NonAffine);
        }
        if degree == 0 {
            has_constant = true;
        } else {
            let slot = highest_set_bit(&term.mask).ok_or(ComputeFailure::Internal)?;
            let entry = singleton_terms
                .get_mut(slot)
                .ok_or(ComputeFailure::Internal)?;
            *entry = Some(term.matrix.as_slice());
        }
    }
    if !has_constant {
        return Err(ComputeFailure::Internal);
    }

    // I08 emits cut widths/live counts only. The reference Packet also forms
    // the target RHS and tests feasibility, but neither value is an I08 channel.
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(slots)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    for matrix in singleton_terms {
        let Some(matrix) = matrix else {
            columns.push(zeroed_u64(flat_words)?);
            continue;
        };
        columns.push(super::rank::flatten_matrix(
            matrix,
            n,
            words_per_row,
            flat_words,
        )?);
    }

    Ok(columns)
}
