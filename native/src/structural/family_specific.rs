use crate::features::FeatureValue;
use crate::schema::FeatureId;
use crate::state::MatrixState;
use std::collections::HashMap;

use super::math::gf2_rank;
use super::matrix::StructuralMatrix;
use super::output::{id, put_global_values, put_pair};

pub(super) fn compute(
    state: &MatrixState,
    data: &StructuralMatrix,
    enabled_ids: &[FeatureId],
    output: &mut HashMap<FeatureId, FeatureValue>,
) -> Result<(), String> {
    let n = data.n;
    let cells = data.cells;
    let row_intersection = &data.row_intersection;
    let row_weight = &data.row_weight;
    let has = |family, ordinal| enabled_ids.contains(&id(family, ordinal));
    if has('C', 1) {
        let mut residual_packed = state.packed_rows().to_vec();
        for (row, words) in residual_packed
            .chunks_exact_mut(state.words_per_row())
            .enumerate()
        {
            words[row / 64] ^= 1u64 << (row % 64);
        }
        put_global_values(
            output,
            'C',
            1,
            1,
            vec![gf2_rank(&residual_packed, n) as f32],
        );
    }

    if has('D', 3) {
        let mut d03 = Vec::with_capacity(cells * 3);
        for i in 0..n {
            for j in 0..n {
                // Source contract: ordered pairs of physical rows, binned by
                // column pattern 10, 01, 11.
                let shared = row_intersection[i * n + j];
                d03.extend([
                    (row_weight[i] - shared) as f32,
                    (row_weight[j] - shared) as f32,
                    shared as f32,
                ]);
            }
        }
        put_pair(output, 'D', 3, n, 3, d03);
    }

    Ok(())
}
