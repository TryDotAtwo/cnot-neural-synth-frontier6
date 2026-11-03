use crate::features::FeatureValue;
use crate::schema::FeatureId;
use std::collections::HashMap;

use super::math::{
    dag_depth, equivalence_classes, hadamard, matmul, row_sums, transitive_closure,
    transpose_dense, undirected_adjacency, undirected_components,
};
use super::matrix::StructuralMatrix;
use super::output::{put_global, put_global_values, put_node, put_pair};

pub(super) fn compute(
    data: &StructuralMatrix,
    output: &mut HashMap<FeatureId, FeatureValue>,
) -> Result<(), String> {
    let n = data.n;
    let cells = data.cells;
    let matrix = &data.matrix;
    let adjacency = &data.adjacency;
    let transpose = &data.transpose;
    let row_weight = &data.row_weight;
    let col_weight = &data.col_weight;
    let components = undirected_components(&adjacency, n);
    let mut component_size = vec![0i64; n];
    let mut component_stats = vec![0i64; n * 3];
    let diagonal_residual = (0..n).map(|i| 1 - matrix[i * n + i]).collect::<Vec<_>>();
    for component in components.iter() {
        let size = component.len() as i64;
        let mut diagonal_sum = 0i64;
        let mut rows_sum = 0i64;
        let mut cols_sum = 0i64;
        for &node in component {
            component_size[node] = size;
            diagonal_sum += diagonal_residual[node];
            rows_sum += row_weight[node];
            cols_sum += col_weight[node];
        }
        for &node in component {
            component_stats[node * 3] = diagonal_sum;
            component_stats[node * 3 + 1] = rows_sum;
            component_stats[node * 3 + 2] = cols_sum;
        }
    }
    put_global(output, 'B', 1, components.len() as i64, 1);
    put_node(
        output,
        'B',
        2,
        n,
        1,
        component_size.iter().map(|v| *v as f32).collect(),
    );

    let reachability = transitive_closure(&adjacency, n);
    let mut mutual = vec![0u8; cells];
    for i in 0..n {
        for j in 0..n {
            mutual[i * n + j] = if reachability[i * n + j] && reachability[j * n + i] {
                1
            } else {
                0
            };
        }
    }
    let sccs = equivalence_classes(&mutual, n);
    let mut scc_label = vec![0usize; n];
    for (label, members) in sccs.iter().enumerate() {
        for &node in members {
            scc_label[node] = label;
        }
    }
    let mut b03 = vec![0f32; n];
    for i in 0..n {
        b03[i] = mutual[i * n..(i + 1) * n]
            .iter()
            .map(|v| f32::from(*v))
            .sum();
    }
    put_node(output, 'B', 3, n, 1, b03);

    let mut dag = vec![0u8; sccs.len() * sccs.len()];
    for i in 0..n {
        for j in 0..n {
            if adjacency[i * n + j] != 0 && scc_label[i] != scc_label[j] {
                dag[scc_label[i] * sccs.len() + scc_label[j]] = 1;
            }
        }
    }
    let depth = dag_depth(&dag, sccs.len());
    let mut b04 = Vec::with_capacity(n * 3);
    for node in 0..n {
        let label = scc_label[node];
        let base = label * sccs.len();
        let indegree = (0..sccs.len())
            .map(|i| dag[i * sccs.len() + label] as i64)
            .sum::<i64>();
        let outdegree = dag[base..base + sccs.len()]
            .iter()
            .map(|v| i64::from(*v))
            .sum::<i64>();
        b04.extend([depth[label] as f32, indegree as f32, outdegree as f32]);
    }
    put_node(output, 'B', 4, n, 3, b04);

    let reciprocal = hadamard(&adjacency, &transpose, n);
    put_node(
        output,
        'B',
        5,
        n,
        1,
        row_sums(&reciprocal, n).iter().map(|v| *v as f32).collect(),
    );
    let paths = matmul(&adjacency, &adjacency, n)?;
    put_pair(
        output,
        'B',
        6,
        n,
        1,
        paths.iter().map(|v| *v as f32).collect(),
    );
    put_pair(
        output,
        'B',
        7,
        n,
        1,
        paths.iter().map(|v| (*v % 2) as f32).collect(),
    );
    // Python B08 uses the diagonal-free adjacency on both sides.
    let adjacency_transpose = transpose_dense(&adjacency, n);
    let at_a = matmul(&adjacency_transpose, &adjacency, n)?;
    let a_at = matmul(&adjacency, &adjacency_transpose, n)?;
    let mut b08 = Vec::with_capacity(cells * 2);
    for i in 0..cells {
        b08.push(at_a[i] as f32);
        b08.push(a_at[i] as f32);
    }
    put_pair(output, 'B', 8, n, 2, b08);
    let paths3 = matmul(&paths, &adjacency, n)?;
    put_node(
        output,
        'B',
        9,
        n,
        1,
        (0..n).map(|i| paths3[i * n + i] as f32).collect(),
    );

    let undirected = undirected_adjacency(&adjacency, n);
    let common_neighbors = matmul(&undirected, &undirected, n)?;
    let mut b10 = Vec::with_capacity(n * 2);
    for i in 0..n {
        let triangles = (0..n)
            .map(|j| common_neighbors[i * n + j] * undirected[i * n + j])
            .sum::<i64>()
            / 2;
        let four_cycles = (0..n)
            .filter(|j| *j != i)
            .map(|j| {
                let common = common_neighbors[i * n + j];
                common * (common - 1) / 2
            })
            .sum::<i64>();
        b10.extend([triangles as f32, four_cycles as f32]);
    }
    put_node(output, 'B', 10, n, 2, b10);
    put_node(
        output,
        'B',
        11,
        n,
        3,
        component_stats.iter().map(|v| *v as f32).collect(),
    );

    let mut b12 = Vec::with_capacity(16);
    for weights in [&row_weight, &col_weight] {
        let mean = weights.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        let variance = weights
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        b12.extend([
            mean as f32,
            variance as f32,
            *weights.iter().max().unwrap_or(&0) as f32,
        ]);
        for bucket in 0..5i64 {
            b12.push(weights.iter().filter(|value| **value == bucket).count() as f32);
        }
    }
    put_global_values(output, 'B', 12, 16, b12);
    Ok(())
}
