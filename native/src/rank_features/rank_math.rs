//! Checked GF(2) projection and packed-row elimination kernels.

use crate::state::MatrixState;

pub(super) fn projected_rank(
    state: &MatrixState,
    selected_rows: &[u32],
    selected_columns: &[u32],
    subtract_identity: bool,
    transpose: bool,
    appended_candidate: Option<&[u32]>,
) -> Result<usize, &'static str> {
    let n = state.dimension() as usize;
    if selected_rows.iter().any(|row| *row as usize >= n)
        || selected_columns.iter().any(|column| *column as usize >= n)
        || selected_rows.is_empty()
        || selected_columns.is_empty()
    {
        return Err("labeled cut index is outside the state");
    }
    if appended_candidate.is_some_and(|support| support.iter().any(|wire| *wire as usize >= n)) {
        return Err("candidate support is outside the state");
    }
    let columns = selected_columns.len();
    let words = columns.checked_add(63).ok_or("cut matrix width overflow")? / 64;
    let row_count = selected_rows
        .len()
        .checked_add(usize::from(appended_candidate.is_some()))
        .ok_or("cut matrix row count overflow")?;
    let word_count = row_count
        .checked_mul(words)
        .ok_or("cut matrix storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "cut matrix workspace allocation failed")?;
    matrix.resize(word_count, 0u64);

    let packed = state.packed_rows();
    for (output_row, selected_row) in selected_rows.iter().copied().enumerate() {
        for (output_column, selected_column) in selected_columns.iter().copied().enumerate() {
            let (source_row, source_column) = if transpose {
                (selected_column as usize, selected_row as usize)
            } else {
                (selected_row as usize, selected_column as usize)
            };
            let source_word = source_row
                .checked_mul(state.words_per_row())
                .and_then(|base| base.checked_add(source_column / 64))
                .ok_or("source matrix index overflow")?;
            if source_word >= packed.len() {
                return Err("labeled cut index is outside the state");
            }
            let mut bit = (packed[source_word] >> (source_column % 64)) & 1;
            if subtract_identity && source_row == source_column {
                bit ^= 1;
            }
            if bit != 0 {
                matrix[output_row * words + output_column / 64] |= 1u64 << (output_column % 64);
            }
        }
    }

    if let Some(candidate_support) = appended_candidate {
        let output_row = selected_rows.len();
        let row_offset = output_row
            .checked_mul(words)
            .ok_or("candidate row index overflow")?;
        for (output_column, source_column) in selected_columns.iter().copied().enumerate() {
            if candidate_support.binary_search(&source_column).is_ok() {
                matrix[row_offset + output_column / 64] |= 1u64 << (output_column % 64);
            }
        }
    }

    Ok(eliminate_packed_rows(
        &mut matrix,
        row_count,
        columns,
        words,
    ))
}

pub(super) fn rank_binary_block(block: &[Vec<u8>]) -> Result<usize, &'static str> {
    let Some(first_row) = block.first() else {
        return Err("partial-shear block has no target rows");
    };
    let columns = first_row.len();
    if columns == 0
        || block
            .iter()
            .any(|row| row.len() != columns || row.iter().any(|bit| *bit > 1))
    {
        return Err("partial-shear block has invalid binary shape");
    }
    let words = columns
        .checked_add(63)
        .ok_or("partial-shear width overflow")?
        / 64;
    let word_count = block
        .len()
        .checked_mul(words)
        .ok_or("partial-shear rank storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "partial-shear rank workspace allocation failed")?;
    matrix.resize(word_count, 0u64);
    for (row_index, row) in block.iter().enumerate() {
        for (column, bit) in row.iter().copied().enumerate() {
            if bit != 0 {
                matrix[row_index * words + column / 64] |= 1u64 << (column % 64);
            }
        }
    }
    Ok(eliminate_packed_rows(
        &mut matrix,
        block.len(),
        columns,
        words,
    ))
}

fn eliminate_packed_rows(
    matrix: &mut [u64],
    row_count: usize,
    columns: usize,
    words: usize,
) -> usize {
    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (rank..row_count).find(|row| matrix[*row * words + word] & bit != 0)
        else {
            continue;
        };
        if pivot != rank {
            for index in 0..words {
                matrix.swap(pivot * words + index, rank * words + index);
            }
        }
        for row in (rank + 1)..row_count {
            if matrix[row * words + word] & bit != 0 {
                for index in 0..words {
                    let pivot_value = matrix[rank * words + index];
                    matrix[row * words + index] ^= pivot_value;
                }
            }
        }
        rank += 1;
        if rank == row_count {
            break;
        }
    }
    rank
}
