use crate::features::FeatureValue;
use crate::schema::FeatureId;
use std::collections::HashMap;

use super::matrix::StructuralMatrix;
use super::output::{id, put_global_values, put_pair};

pub(super) fn compute(
    data: &StructuralMatrix,
    enabled_ids: &[FeatureId],
    output: &mut HashMap<FeatureId, FeatureValue>,
) -> Result<(), String> {
    let n = data.n;
    let cells = data.cells;
    let col_intersection = &data.col_intersection;
    let col_weight = &data.col_weight;
    let residual_row_weight = &data.residual_row_weight;
    let residual_col_weight = &data.residual_col_weight;
    put_global_values(
        output,
        'F',
        1,
        2,
        vec![
            residual_row_weight
                .iter()
                .filter(|value| **value > 0)
                .count() as f32,
            residual_col_weight
                .iter()
                .filter(|value| **value > 0)
                .count() as f32,
        ],
    );
    if enabled_ids.contains(&id('G', 5)) {
        let Some(value_count) = cells.checked_mul(4) else {
            return Err("G05 pair-histogram output size overflow".into());
        };
        let mut g05 = Vec::new();
        g05.try_reserve_exact(value_count)
            .map_err(|_| "G05 pair-histogram output allocation failed")?;
        for left in 0..n {
            for right in 0..n {
                let overlap = col_intersection[left * n + right];
                let left_weight = col_weight[left];
                let right_weight = col_weight[right];
                g05.extend([
                    (n as i64 - left_weight - right_weight + overlap) as f32,
                    (left_weight - overlap) as f32,
                    (right_weight - overlap) as f32,
                    overlap as f32,
                ]);
            }
        }
        put_pair(output, 'G', 5, n, 4, g05);
    }

    Ok(())
}
