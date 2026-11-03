//! Incremental GF(2) rank bases for ordered affine packet columns.

use super::storage::{bit, highest_set_bit, set_bit, xor_assign, zeroed_u64};
use super::ComputeFailure;
use std::collections::BTreeMap;

pub(super) fn incremental_ranks(
    columns: &[Vec<u64>],
    reverse: bool,
) -> Result<Vec<usize>, ComputeFailure> {
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

pub(super) fn flatten_matrix(
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
