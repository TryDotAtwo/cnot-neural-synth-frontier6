use crate::state::MatrixState;

use super::math::{col_sums, matmul, row_sums, transpose_dense, zeroed};

pub(super) struct StructuralMatrix {
    pub(super) n: usize,
    pub(super) cells: usize,
    pub(super) matrix: Vec<i64>,
    pub(super) residual: Vec<i64>,
    pub(super) adjacency: Vec<i64>,
    pub(super) transpose: Vec<i64>,
    pub(super) row_weight: Vec<i64>,
    pub(super) col_weight: Vec<i64>,
    pub(super) residual_row_weight: Vec<i64>,
    pub(super) residual_col_weight: Vec<i64>,
    pub(super) row_intersection: Vec<i64>,
    pub(super) col_intersection: Vec<i64>,
    pub(super) row_difference: Vec<i64>,
    pub(super) col_difference: Vec<i64>,
}

impl StructuralMatrix {
    pub(super) fn from_state(state: &MatrixState) -> Result<Self, String> {
        let n = usize::try_from(state.dimension()).map_err(|_| "dimension does not fit usize")?;
        let cells = n.checked_mul(n).ok_or("matrix size overflow")?;
        let mut matrix = zeroed(cells)?;
        for row in 0..n {
            let words = &state.packed_rows()
                [row * state.words_per_row()..(row + 1) * state.words_per_row()];
            for col in 0..n {
                matrix[row * n + col] = ((words[col / 64] >> (col % 64)) & 1) as i64;
            }
        }

        let mut residual = matrix.clone();
        let mut adjacency = matrix.clone();
        for i in 0..n {
            residual[i * n + i] ^= 1;
            adjacency[i * n + i] = 0;
        }
        let transpose = transpose_dense(&matrix, n);
        let row_weight = row_sums(&matrix, n);
        let col_weight = col_sums(&matrix, n);
        let residual_row_weight = row_sums(&residual, n);
        let residual_col_weight = col_sums(&residual, n);
        let row_intersection = matmul(&matrix, &transpose, n)?;
        let col_intersection = matmul(&transpose, &matrix, n)?;
        let row_difference = (0..cells)
            .map(|index| row_weight[index / n] - row_intersection[index])
            .collect::<Vec<_>>();
        let col_difference = (0..cells)
            .map(|index| col_weight[index / n] - col_intersection[index])
            .collect::<Vec<_>>();

        Ok(Self {
            n,
            cells,
            matrix,
            residual,
            adjacency,
            transpose,
            row_weight,
            col_weight,
            residual_row_weight,
            residual_col_weight,
            row_intersection,
            col_intersection,
            row_difference,
            col_difference,
        })
    }
}
