use crate::state::MatrixState;

use super::TrainingDataError;

/// max(changed rows, changed columns, rank(M-I)). Over GF(2) the rank is
/// bounded by both support counts, so their maximum is already the exact result.
pub fn elementary_lower_bound(state: &MatrixState) -> Result<u64, TrainingDataError> {
    let width = state.words_per_row();
    let mut columns = vec![0u64; width];
    let mut changed_rows = 0u64;
    for (i, row) in state.packed_rows().chunks_exact(width).enumerate() {
        let mut nonzero = false;
        for (j, (&packed, column)) in row.iter().zip(&mut columns).enumerate() {
            let identity = if j == i / 64 { 1u64 << (i % 64) } else { 0 };
            let residual = packed ^ identity;
            nonzero |= residual != 0;
            *column |= residual;
        }
        changed_rows += u64::from(nonzero);
    }
    let changed_columns = columns.iter().map(|w| u64::from(w.count_ones())).sum();
    Ok(changed_rows.max(changed_columns))
}
