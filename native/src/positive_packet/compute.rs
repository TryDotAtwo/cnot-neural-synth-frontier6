use super::{LocalBooleanFactor, MAX_EXACT_F32_INTEGER, OptionalGatePacket};
use crate::features::FeatureValue;
use crate::state::MatrixState;
use std::collections::BTreeMap;
use std::mem::size_of;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ComputeFailure {
    ResourceLimit,
    NonAffine,
    Internal,
}

#[derive(Clone, Debug)]
struct PolynomialTerm {
    mask: Vec<u64>,
    matrix: Vec<u64>,
}

pub(super) fn compute_values(
    state: &MatrixState,
    context: &OptionalGatePacket,
    max_monomials: usize,
) -> Result<FeatureValue, ComputeFailure> {
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
        columns.push(flatten_matrix(matrix, n, words_per_row, flat_words)?);
    }

    let prefix = incremental_ranks(&columns, false)?;
    let suffix = incremental_ranks(&columns, true)?;
    let total_rank = *prefix.last().ok_or(ComputeFailure::Internal)?;
    let groups = slots.checked_add(1).ok_or(ComputeFailure::ResourceLimit)?;
    let value_count = groups.checked_mul(4).ok_or(ComputeFailure::ResourceLimit)?;
    let incidence_count = groups
        .checked_mul(n)
        .and_then(|count| count.checked_mul(2))
        .ok_or(ComputeFailure::ResourceLimit)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    let mut incidence = zeroed_f32(incidence_count)?;
    let mut token_valid = Vec::new();
    token_valid
        .try_reserve_exact(groups)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    token_valid.resize(groups, 1);

    for cut in 0..groups {
        let width = prefix[cut]
            .checked_add(suffix[cut])
            .and_then(|sum| sum.checked_sub(total_rank))
            .ok_or(ComputeFailure::Internal)?;
        let mut live_slots = Vec::new();
        live_slots
            .try_reserve_exact(cut)
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        for slot in 0..cut {
            let live = context.factors.iter().any(|factor| {
                factor.scope.iter().copied().max().is_some_and(|last| {
                    usize::try_from(last).is_ok_and(|last| last >= cut)
                        && factor.scope.contains(&(slot as u32))
                })
            });
            if live {
                live_slots.push(slot);
                let (control, target) = context.gates[slot];
                let control = usize::try_from(control).map_err(|_| ComputeFailure::Internal)?;
                let target = usize::try_from(target).map_err(|_| ComputeFailure::Internal)?;
                incidence[(cut * n + control) * 2] = 1.0;
                incidence[(cut * n + target) * 2 + 1] = 1.0;
            }
        }
        let live = live_slots.len();
        let upper = width.checked_add(live).ok_or(ComputeFailure::Internal)?;
        if cut > MAX_EXACT_F32_INTEGER || upper > MAX_EXACT_F32_INTEGER {
            return Err(ComputeFailure::ResourceLimit);
        }
        values.extend([cut as f32, width as f32, live as f32, upper as f32]);
    }

    // Keep the state argument in the exact contract even though the displayed
    // widths are independent of its RHS.
    if state.dimension() != context.n {
        return Err(ComputeFailure::Internal);
    }
    Ok(FeatureValue {
        shape: vec![groups, 4],
        values,
        token_valid,
        incidence,
    })
}

pub(super) fn working_bytes_upper_bound(
    n: usize,
    slots: usize,
    max_monomials: usize,
) -> Option<usize> {
    let words_per_row = n.checked_add(63)?.checked_div(64)?;
    let matrix_bytes = n
        .checked_mul(words_per_row)?
        .checked_mul(size_of::<u64>())?;
    let mask_bytes = words_for(slots)?.checked_mul(size_of::<u64>())?;
    let per_term = matrix_bytes
        .checked_add(mask_bytes)?
        .checked_add(size_of::<PolynomialTerm>())?
        .checked_add(128)?;
    let polynomial = max_monomials.checked_mul(per_term)?.checked_mul(2)?;
    let flat_words = words_for(n.checked_mul(n)?)?;
    let rank_workspace = slots
        .checked_mul(flat_words)?
        .checked_mul(size_of::<u64>())?
        .checked_mul(2)?;
    let groups = slots.checked_add(1)?;
    let output = groups
        .checked_mul(
            4usize
                .checked_add(n.checked_mul(2)?)?
                .checked_mul(size_of::<f32>())?,
        )?
        .checked_add(groups)?;
    polynomial.checked_add(rank_workspace)?.checked_add(output)
}

fn incremental_ranks(columns: &[Vec<u64>], reverse: bool) -> Result<Vec<usize>, ComputeFailure> {
    let mut ranks = Vec::new();
    ranks
        .try_reserve_exact(
            columns
                .len()
                .checked_add(1)
                .ok_or(ComputeFailure::ResourceLimit)?,
        )
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    ranks.resize(columns.len() + 1, 0);
    let mut pivots = BTreeMap::<usize, Vec<u64>>::new();
    if reverse {
        for index in (0..columns.len()).rev() {
            insert_basis(&mut pivots, &columns[index])?;
            ranks[index] = pivots.len();
        }
    } else {
        for index in 0..columns.len() {
            insert_basis(&mut pivots, &columns[index])?;
            ranks[index + 1] = pivots.len();
        }
    }
    Ok(ranks)
}

fn insert_basis(
    pivots: &mut BTreeMap<usize, Vec<u64>>,
    vector: &[u64],
) -> Result<(), ComputeFailure> {
    let mut row = Vec::new();
    row.try_reserve_exact(vector.len())
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    row.extend_from_slice(vector);
    for (pivot, basis_row) in pivots.iter().rev() {
        if bit(&row, *pivot) {
            xor_assign(&mut row, basis_row);
        }
    }
    let Some(pivot) = highest_set_bit(&row) else {
        return Ok(());
    };
    for basis_row in pivots.values_mut() {
        if bit(basis_row, pivot) {
            xor_assign(basis_row, &row);
        }
    }
    pivots.insert(pivot, row);
    Ok(())
}

fn flatten_matrix(
    matrix: &[u64],
    n: usize,
    words_per_row: usize,
    flat_words: usize,
) -> Result<Vec<u64>, ComputeFailure> {
    let mut flat = zeroed_u64(flat_words)?;
    for row in 0..n {
        for source_word in 0..words_per_row {
            let mut word = matrix[row * words_per_row + source_word];
            while word != 0 {
                let column = source_word * 64 + word.trailing_zeros() as usize;
                if column < n {
                    let bit_index = row
                        .checked_mul(n)
                        .and_then(|offset| offset.checked_add(column))
                        .ok_or(ComputeFailure::ResourceLimit)?;
                    set_bit(&mut flat, bit_index);
                }
                word &= word - 1;
            }
        }
    }
    Ok(flat)
}

fn words_for(width: usize) -> Option<usize> {
    width.checked_add(63).map(|value| value / 64)
}

fn zeroed_u64(length: usize) -> Result<Vec<u64>, ComputeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    values.resize(length, 0);
    Ok(values)
}

fn zeroed_f32(length: usize) -> Result<Vec<f32>, ComputeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn set_bit(words: &mut [u64], index: usize) {
    words[index / 64] |= 1u64 << (index % 64);
}

fn bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

fn highest_set_bit(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().rev().find_map(|(index, word)| {
        (*word != 0).then(|| index * 64 + (63 - word.leading_zeros() as usize))
    })
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= right;
    }
}

pub(super) fn factor_entry_count(factors: &[LocalBooleanFactor]) -> Option<usize> {
    factors.iter().try_fold(0usize, |total, factor| {
        total
            .checked_add(factor.scope.len())?
            .checked_add(factor.allowed.len())
    })
}
