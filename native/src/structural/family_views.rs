use crate::features::FeatureValue;
use crate::schema::FeatureId;
use crate::state::MatrixState;
use std::collections::HashMap;

use super::math::{gf2_inverse, packed_to_dense, transpose_dense};
use super::matrix::StructuralMatrix;
use super::output::{id, put_pair};

pub(super) fn compute(
    state: &MatrixState,
    data: &StructuralMatrix,
    enabled_ids: &[FeatureId],
    output: &mut HashMap<FeatureId, FeatureValue>,
) -> Result<(), String> {
    let n = data.n;
    let cells = data.cells;
    let matrix = &data.matrix;
    let transpose = &data.transpose;
    let has = |family, ordinal| enabled_ids.contains(&id(family, ordinal));
    if has('H', 1) {
        put_pair(
            output,
            'H',
            1,
            n,
            1,
            matrix.iter().map(|v| *v as f32).collect(),
        );
    }
    if has('H', 2) {
        put_pair(
            output,
            'H',
            2,
            n,
            1,
            transpose.iter().map(|v| *v as f32).collect(),
        );
    }
    if has('H', 3) || has('H', 4) || has('H', 5) {
        let inverse_packed = gf2_inverse(state.packed_rows(), n)?;
        let inverse_dense = packed_to_dense(&inverse_packed, n);
        let inverse_transpose = if has('H', 4) || has('H', 5) {
            Some(transpose_dense(&inverse_dense, n))
        } else {
            None
        };
        if has('H', 3) {
            put_pair(
                output,
                'H',
                3,
                n,
                1,
                inverse_dense.iter().map(|v| *v as f32).collect(),
            );
        }
        if has('H', 4) {
            put_pair(
                output,
                'H',
                4,
                n,
                1,
                inverse_transpose
                    .as_ref()
                    .expect("inverse transpose was requested")
                    .iter()
                    .map(|v| *v as f32)
                    .collect(),
            );
        }
        if has('H', 5) {
            // H05 supplies aligned raw pair inputs to the model's learned
            // pair-token projection. It does not claim the four views are
            // interchangeable under an arbitrary CNOT alphabet.
            let value_count = cells
                .checked_mul(4)
                .ok_or("H05 aligned-view output size overflow")?;
            let mut h05 = Vec::new();
            h05.try_reserve_exact(value_count)
                .map_err(|_| "H05 aligned-view output allocation failed")?;
            let inverse_transpose = inverse_transpose
                .as_ref()
                .expect("inverse transpose was requested");
            for index in 0..cells {
                h05.extend([
                    matrix[index] as f32,
                    transpose[index] as f32,
                    inverse_dense[index] as f32,
                    inverse_transpose[index] as f32,
                ]);
            }
            put_pair(output, 'H', 5, n, 4, h05);
        }
    }
    Ok(())
}
