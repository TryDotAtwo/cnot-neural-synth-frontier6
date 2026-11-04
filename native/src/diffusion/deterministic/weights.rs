//! Packed-matrix summaries shared by diffusion projections.

use crate::state::MatrixState;

pub(super) fn row_weights(state: &MatrixState) -> Vec<usize> {
    let words = state.words_per_row() as usize;
    state
        .packed_rows()
        .chunks_exact(words)
        .map(|row| row.iter().map(|word| word.count_ones() as usize).sum())
        .collect()
}

pub(super) fn column_weights(state: &MatrixState) -> Vec<usize> {
    let n = state.dimension() as usize;
    (0..n)
        .map(|column| {
            (0..n)
                .map(|row| matrix_bit(state, row, column) as usize)
                .sum()
        })
        .collect()
}

pub(super) fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> bool {
    let words_per_row = state.words_per_row() as usize;
    let word = state.packed_rows()[row * words_per_row + column / 64];
    ((word >> (column % 64)) & 1) != 0
}

pub(super) fn pair_histogram(state: &MatrixState, first: usize, second: usize) -> [usize; 4] {
    let n = state.dimension() as usize;
    let mut histogram = [0usize; 4];
    for row in 0..n {
        let pattern = (matrix_bit(state, row, first) as usize)
            | ((matrix_bit(state, row, second) as usize) << 1);
        histogram[pattern] += 1;
    }
    histogram
}
