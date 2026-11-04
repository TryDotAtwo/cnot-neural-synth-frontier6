use crate::features::FeatureValue;
use crate::schema::FeatureId;
use std::collections::HashMap;

use super::math::matmul;
use super::matrix::StructuralMatrix;
use super::output::{put_node, put_pair};

pub(super) fn compute(
    data: &StructuralMatrix,
    output: &mut HashMap<FeatureId, FeatureValue>,
) -> Result<(), String> {
    let n = data.n;
    let cells = data.cells;
    let matrix = &data.matrix;
    let residual = &data.residual;
    let transpose = &data.transpose;
    let row_weight = &data.row_weight;
    let col_weight = &data.col_weight;
    let residual_row_weight = &data.residual_row_weight;
    let residual_col_weight = &data.residual_col_weight;
    let row_intersection = &data.row_intersection;
    let col_intersection = &data.col_intersection;
    let row_difference = &data.row_difference;
    let col_difference = &data.col_difference;
    put_pair(
        output,
        'A',
        1,
        n,
        1,
        matrix.iter().map(|v| *v as f32).collect(),
    );
    put_pair(
        output,
        'A',
        2,
        n,
        1,
        transpose.iter().map(|v| *v as f32).collect(),
    );
    let mut identity_ones = Vec::with_capacity(cells * 2);
    for i in 0..n {
        for j in 0..n {
            identity_ones.push(if i == j { 1.0 } else { 0.0 });
            identity_ones.push(1.0);
        }
    }
    put_pair(output, 'A', 3, n, 2, identity_ones);
    put_pair(
        output,
        'A',
        4,
        n,
        1,
        residual.iter().map(|v| *v as f32).collect(),
    );

    let mut a05 = Vec::with_capacity(n * 2);
    for value in row_weight {
        a05.push(*value as f32);
        a05.push(*value as f32 / n as f32);
    }
    put_node(output, 'A', 5, n, 2, a05);
    put_node(
        output,
        'A',
        6,
        n,
        1,
        col_weight.iter().map(|v| *v as f32).collect(),
    );
    put_node(
        output,
        'A',
        7,
        n,
        1,
        residual_row_weight.iter().map(|v| *v as f32).collect(),
    );
    put_node(
        output,
        'A',
        8,
        n,
        1,
        residual_col_weight.iter().map(|v| *v as f32).collect(),
    );
    put_node(
        output,
        'A',
        9,
        n,
        1,
        residual_row_weight
            .iter()
            .map(|v| if *v > 0 { 1.0 } else { 0.0 })
            .collect(),
    );
    put_node(
        output,
        'A',
        10,
        n,
        1,
        residual_col_weight
            .iter()
            .map(|v| if *v > 0 { 1.0 } else { 0.0 })
            .collect(),
    );
    put_node(
        output,
        'A',
        11,
        n,
        1,
        (0..n).map(|i| (1 - matrix[i * n + i]) as f32).collect(),
    );
    let mut a12 = Vec::with_capacity(n * 2);
    for i in 0..n {
        a12.push((row_weight[i] & 1) as f32);
        a12.push((col_weight[i] & 1) as f32);
    }
    put_node(output, 'A', 12, n, 2, a12);

    put_pair(
        output,
        'A',
        13,
        n,
        1,
        row_intersection.iter().map(|v| *v as f32).collect(),
    );
    put_pair(
        output,
        'A',
        14,
        n,
        1,
        col_intersection.iter().map(|v| *v as f32).collect(),
    );
    let mut a15 = Vec::with_capacity(cells);
    let mut a16 = Vec::with_capacity(cells);
    for i in 0..n {
        for j in 0..n {
            a15.push((row_weight[i] + row_weight[j] - 2 * row_intersection[i * n + j]) as f32);
            a16.push((col_weight[i] + col_weight[j] - 2 * col_intersection[i * n + j]) as f32);
        }
    }
    put_pair(output, 'A', 15, n, 1, a15);
    put_pair(output, 'A', 16, n, 1, a16);
    let mut a17 = Vec::with_capacity(cells * 2);
    let mut a18 = Vec::with_capacity(cells * 2);
    let mut a19 = Vec::with_capacity(cells * 2);
    for i in 0..n {
        for j in 0..n {
            let index = i * n + j;
            a17.push(row_difference[index] as f32);
            a17.push(row_difference[j * n + i] as f32);
            a18.push(col_difference[index] as f32);
            a18.push(col_difference[j * n + i] as f32);
            a19.push(if row_difference[index] == 0 { 1.0 } else { 0.0 });
            a19.push(if col_difference[index] == 0 { 1.0 } else { 0.0 });
        }
    }
    put_pair(output, 'A', 17, n, 2, a17);
    put_pair(output, 'A', 18, n, 2, a18);
    put_pair(output, 'A', 19, n, 2, a19);

    let residual_row_dot_matrix_transpose = matmul(residual, transpose, n)?;
    let mut a20 = Vec::with_capacity(cells * 2);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                a20.extend([0.0, 0.0]);
            } else {
                let first = row_weight[i] - 2 * row_intersection[i * n + j];
                // Source formula: (E @ M.T).T[i,j] = E[j] dot M[i].
                let second = row_weight[i] - 2 * residual_row_dot_matrix_transpose[j * n + i];
                a20.extend([first as f32, second as f32]);
            }
        }
    }
    put_pair(output, 'A', 20, n, 2, a20);

    Ok(())
}
