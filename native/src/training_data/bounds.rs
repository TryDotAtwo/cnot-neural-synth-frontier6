use crate::state::MatrixState;

use super::TrainingDataError;

/// Port of the source `cheap_bounds` function: nonzero rows/columns and GF(2)
/// rank of `M xor I`, combined by their maximum.
pub fn elementary_lower_bound(state: &MatrixState) -> Result<u64, TrainingDataError> {
    let n = usize::try_from(state.dimension())
        .map_err(|_| TrainingDataError("matrix dimension exceeds host index range".into()))?;
    let words_per_row = state.words_per_row();
    let mut residual = Vec::new();
    residual
        .try_reserve_exact(n)
        .map_err(|_| TrainingDataError("could not allocate lower-bound rows".into()))?;

    let mut changed_rows = 0usize;
    let mut changed_columns = vec![false; n];
    for row_index in 0..n {
        let start = row_index
            .checked_mul(words_per_row)
            .ok_or_else(|| TrainingDataError("matrix row offset overflow".into()))?;
        let mut row = state.packed_rows()[start..start + words_per_row].to_vec();
        row[row_index / 64] ^= 1u64 << (row_index % 64);
        changed_rows += usize::from(row.iter().any(|word| *word != 0));
        for (column, changed) in changed_columns.iter_mut().enumerate() {
            if ((row[column / 64] >> (column % 64)) & 1) != 0 {
                *changed = true;
            }
        }
        residual.push(row);
    }
    let rank = gf2_rank_packed(&mut residual, n);
    let changed_columns = changed_columns
        .into_iter()
        .filter(|changed| *changed)
        .count();
    Ok(changed_rows.max(changed_columns).max(rank) as u64)
}

fn gf2_rank_packed(rows: &mut [Vec<u64>], n: usize) -> usize {
    let mut pivots: Vec<Option<Vec<u64>>> = vec![None; n];
    let mut rank = 0usize;
    for row in rows {
        for column in (0..n).rev() {
            if ((row[column / 64] >> (column % 64)) & 1) == 0 {
                continue;
            }
            if let Some(pivot) = &pivots[column] {
                for (word, pivot_word) in row.iter_mut().zip(pivot) {
                    *word ^= pivot_word;
                }
            } else {
                pivots[column] = Some(row.clone());
                rank += 1;
                break;
            }
        }
    }
    rank
}
