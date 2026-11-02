//! Exact ports of selected source feature producers.
//!
//! Current built-in coverage is 41 of the canonical 103 IDs: A01-A20,
//! B01-B12, F01, G05, C01, D03 and H01-H05. Contextual providers add
//! coverage only when their explicit physical input contracts are supplied;
//! no placeholder values are emitted for unavailable features.

use crate::features::FeatureValue;
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use crate::state::MatrixState;
use std::collections::HashMap;

pub fn is_builtin(id: FeatureId) -> bool {
    match id.family() {
        'A' | 'B' => true,
        'F' => id.ordinal() == 1,
        'G' => id.ordinal() == 5,
        'C' => id.ordinal() == 1,
        'D' => id.ordinal() == 3,
        'H' => (1..=5).contains(&id.ordinal()),
        _ => false,
    }
}

#[derive(Clone, Copy)]
pub struct BuiltinMetadata {
    pub scope: &'static str,
    pub producer_version: &'static str,
    pub dependencies: &'static [&'static str],
    pub exactness: &'static str,
    pub cost_tier: &'static str,
}

pub fn metadata(id: FeatureId) -> Option<BuiltinMetadata> {
    let (version, cost, dependencies) = match (id.family(), id.ordinal()) {
        ('A', 1..=20) | ('B', 1..=12) | ('F', 1) => ("structural_v1", "T1", &["M"][..]),
        ('G', 5) => (
            "exact_column_pair_histogram_v1",
            "T1",
            &["M", "column_weights", "column_intersections"][..],
        ),
        ('C', 1) => ("gf2_rank_residual_v1", "T1", &["M"][..]),
        ('D', 3) => (
            "width2_parity_multiplicity_v1",
            "T1",
            &["M", "width2_source_pair"][..],
        ),
        ('H', 1) => ("matrix_view_v1", "T0", &["M"][..]),
        ('H', 2) => ("matrix_transpose_view_v1", "T0", &["M"][..]),
        ('H', 3) => ("matrix_inverse_view_v1", "T1", &["M"][..]),
        ('H', 4) => ("matrix_inverse_transpose_view_v1", "T1", &["M"][..]),
        ('H', 5) => (
            "aligned_four_matrix_views_pair_tokens_v1",
            "T2",
            &["M", "transpose", "inverse", "inverse_transpose"][..],
        ),
        _ => return None,
    };
    Some(BuiltinMetadata {
        scope: "full_labeled_matrix",
        producer_version: version,
        dependencies,
        exactness: "exact",
        cost_tier: cost,
    })
}

pub fn compute(
    state: &MatrixState,
    enabled_ids: &[FeatureId],
) -> Result<HashMap<FeatureId, FeatureValue>, String> {
    let n = usize::try_from(state.dimension()).map_err(|_| "dimension does not fit usize")?;
    let cells = n.checked_mul(n).ok_or("matrix size overflow")?;
    let mut matrix = zeroed(cells)?;
    for row in 0..n {
        let words =
            &state.packed_rows()[row * state.words_per_row()..(row + 1) * state.words_per_row()];
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

    let mut output = HashMap::with_capacity(41);
    put_pair(
        &mut output,
        'A',
        1,
        n,
        1,
        matrix.iter().map(|v| *v as f32).collect(),
    );
    put_pair(
        &mut output,
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
    put_pair(&mut output, 'A', 3, n, 2, identity_ones);
    put_pair(
        &mut output,
        'A',
        4,
        n,
        1,
        residual.iter().map(|v| *v as f32).collect(),
    );

    let mut a05 = Vec::with_capacity(n * 2);
    for value in &row_weight {
        a05.push(*value as f32);
        a05.push(*value as f32 / n as f32);
    }
    put_node(&mut output, 'A', 5, n, 2, a05);
    put_node(
        &mut output,
        'A',
        6,
        n,
        1,
        col_weight.iter().map(|v| *v as f32).collect(),
    );
    put_node(
        &mut output,
        'A',
        7,
        n,
        1,
        residual_row_weight.iter().map(|v| *v as f32).collect(),
    );
    put_node(
        &mut output,
        'A',
        8,
        n,
        1,
        residual_col_weight.iter().map(|v| *v as f32).collect(),
    );
    put_node(
        &mut output,
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
        &mut output,
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
        &mut output,
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
    put_node(&mut output, 'A', 12, n, 2, a12);

    put_pair(
        &mut output,
        'A',
        13,
        n,
        1,
        row_intersection.iter().map(|v| *v as f32).collect(),
    );
    put_pair(
        &mut output,
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
    put_pair(&mut output, 'A', 15, n, 1, a15);
    put_pair(&mut output, 'A', 16, n, 1, a16);
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
    put_pair(&mut output, 'A', 17, n, 2, a17);
    put_pair(&mut output, 'A', 18, n, 2, a18);
    put_pair(&mut output, 'A', 19, n, 2, a19);

    let residual_row_dot_matrix_transpose = matmul(&residual, &transpose, n)?;
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
    put_pair(&mut output, 'A', 20, n, 2, a20);

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
    put_global(&mut output, 'B', 1, components.len() as i64, 1);
    put_node(
        &mut output,
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
    put_node(&mut output, 'B', 3, n, 1, b03);

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
    put_node(&mut output, 'B', 4, n, 3, b04);

    let reciprocal = hadamard(&adjacency, &transpose, n);
    put_node(
        &mut output,
        'B',
        5,
        n,
        1,
        row_sums(&reciprocal, n).iter().map(|v| *v as f32).collect(),
    );
    let paths = matmul(&adjacency, &adjacency, n)?;
    put_pair(
        &mut output,
        'B',
        6,
        n,
        1,
        paths.iter().map(|v| *v as f32).collect(),
    );
    put_pair(
        &mut output,
        'B',
        7,
        n,
        1,
        paths.iter().map(|v| (*v % 2) as f32).collect(),
    );
    let at_a = matmul(&transpose, &adjacency, n)?;
    let a_at = matmul(&adjacency, &transpose, n)?;
    let mut b08 = Vec::with_capacity(cells * 2);
    for i in 0..cells {
        b08.push(at_a[i] as f32);
        b08.push(a_at[i] as f32);
    }
    put_pair(&mut output, 'B', 8, n, 2, b08);
    let paths3 = matmul(&paths, &adjacency, n)?;
    put_node(
        &mut output,
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
    put_node(&mut output, 'B', 10, n, 2, b10);
    put_node(
        &mut output,
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
    put_global_values(&mut output, 'B', 12, 16, b12);
    put_global_values(
        &mut output,
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
        put_pair(&mut output, 'G', 5, n, 4, g05);
    }

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
            &mut output,
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
        put_pair(&mut output, 'D', 3, n, 3, d03);
    }

    if has('H', 1) {
        put_pair(
            &mut output,
            'H',
            1,
            n,
            1,
            matrix.iter().map(|v| *v as f32).collect(),
        );
    }
    if has('H', 2) {
        put_pair(
            &mut output,
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
                &mut output,
                'H',
                3,
                n,
                1,
                inverse_dense.iter().map(|v| *v as f32).collect(),
            );
        }
        if has('H', 4) {
            put_pair(
                &mut output,
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
            put_pair(&mut output, 'H', 5, n, 4, h05);
        }
    }
    Ok(output)
}

fn id(family: char, ordinal: u8) -> FeatureId {
    FeatureId::new(family as u8, ordinal).expect("built-in feature IDs are in the manifest")
}

fn put_node(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    n: usize,
    channels: usize,
    values: Vec<f32>,
) {
    put(
        output,
        family,
        ordinal,
        FeatureRole::Node,
        vec![n, channels],
        values,
    );
}

fn put_pair(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    n: usize,
    channels: usize,
    values: Vec<f32>,
) {
    put(
        output,
        family,
        ordinal,
        FeatureRole::Pair,
        vec![n, n, channels],
        values,
    );
}

fn put_global(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    value: i64,
    channels: usize,
) {
    put_global_values(output, family, ordinal, channels, vec![value as f32]);
}

fn put_global_values(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    channels: usize,
    values: Vec<f32>,
) {
    put(
        output,
        family,
        ordinal,
        FeatureRole::Global,
        vec![channels],
        values,
    );
}

fn put(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    role: FeatureRole,
    shape: Vec<usize>,
    values: Vec<f32>,
) {
    let feature_id = id(family, ordinal);
    debug_assert_eq!(feature_spec(feature_id).role, role);
    output.insert(
        feature_id,
        FeatureValue {
            shape,
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
    );
}

fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("feature allocation failed: {error}"))?;
    values.resize(len, T::default());
    Ok(values)
}

fn gf2_rank(rows: &[u64], n: usize) -> usize {
    let words = n.div_ceil(64);
    let mut matrix = rows.to_vec();
    let mut rank = 0usize;
    for col in 0..n {
        let pivot =
            (rank..n).find(|row| matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0);
        let Some(pivot) = pivot else { continue };
        matrix.swap(rank * words, pivot * words);
        for word in 1..words {
            matrix.swap(rank * words + word, pivot * words + word);
        }
        for row in 0..n {
            if row != rank && matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0 {
                for word in 0..words {
                    matrix[row * words + word] ^= matrix[rank * words + word];
                }
            }
        }
        rank += 1;
        if rank == n {
            break;
        }
    }
    rank
}

pub(crate) fn gf2_inverse(rows: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let words = n.div_ceil(64);
    let mut matrix = rows.to_vec();
    let mut inverse = zeroed(n.checked_mul(words).ok_or("inverse size overflow")?)?;
    for i in 0..n {
        inverse[i * words + i / 64] |= 1u64 << (i % 64);
    }
    for col in 0..n {
        let pivot = (col..n)
            .find(|row| matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0)
            .ok_or("input matrix is singular")?;
        for word in 0..words {
            matrix.swap(col * words + word, pivot * words + word);
            inverse.swap(col * words + word, pivot * words + word);
        }
        for row in 0..n {
            if row != col && matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0 {
                for word in 0..words {
                    matrix[row * words + word] ^= matrix[col * words + word];
                    inverse[row * words + word] ^= inverse[col * words + word];
                }
            }
        }
    }
    Ok(inverse)
}

fn packed_to_dense(rows: &[u64], n: usize) -> Vec<i64> {
    let words = n.div_ceil(64);
    (0..n * n)
        .map(|index| {
            ((rows[(index / n) * words + (index % n) / 64] >> ((index % n) % 64)) & 1) as i64
        })
        .collect()
}

fn transpose_dense(matrix: &[i64], n: usize) -> Vec<i64> {
    let mut result = vec![0; matrix.len()];
    for i in 0..n {
        for j in 0..n {
            result[j * n + i] = matrix[i * n + j];
        }
    }
    result
}

fn row_sums(matrix: &[i64], n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| matrix[i * n..(i + 1) * n].iter().sum())
        .collect()
}

fn col_sums(matrix: &[i64], n: usize) -> Vec<i64> {
    (0..n)
        .map(|j| (0..n).map(|i| matrix[i * n + j]).sum())
        .collect()
}

fn matmul(left: &[i64], right: &[i64], n: usize) -> Result<Vec<i64>, String> {
    let cells = n.checked_mul(n).ok_or("matrix product size overflow")?;
    let mut result = zeroed(cells)?;
    for i in 0..n {
        for k in 0..n {
            let value = left[i * n + k];
            if value == 0 {
                continue;
            }
            for j in 0..n {
                result[i * n + j] += value * right[k * n + j];
            }
        }
    }
    Ok(result)
}

fn hadamard(left: &[i64], right: &[i64], n: usize) -> Vec<i64> {
    (0..n * n).map(|i| left[i] * right[i]).collect()
}

fn undirected_adjacency(adjacency: &[i64], n: usize) -> Vec<i64> {
    (0..n * n)
        .map(|i| {
            if adjacency[i] != 0 || adjacency[(i % n) * n + i / n] != 0 {
                1
            } else {
                0
            }
        })
        .collect()
}

fn undirected_components(adjacency: &[i64], n: usize) -> Vec<Vec<usize>> {
    let undirected = undirected_adjacency(adjacency, n);
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut component = Vec::new();
        let mut queue = vec![root];
        let mut cursor = 0;
        while cursor < queue.len() {
            let node = queue[cursor];
            cursor += 1;
            component.push(node);
            for next in 0..n {
                if undirected[node * n + next] != 0 && !seen[next] {
                    seen[next] = true;
                    queue.push(next);
                }
            }
        }
        components.push(component);
    }
    components
}

fn transitive_closure(adjacency: &[i64], n: usize) -> Vec<bool> {
    let mut reach = (0..n * n)
        .map(|i| adjacency[i] != 0 || i / n == i % n)
        .collect::<Vec<_>>();
    for k in 0..n {
        for i in 0..n {
            if reach[i * n + k] {
                for j in 0..n {
                    reach[i * n + j] |= reach[k * n + j];
                }
            }
        }
    }
    reach
}

fn equivalence_classes(mutual: &[u8], n: usize) -> Vec<Vec<usize>> {
    let mut remaining = vec![true; n];
    let mut classes = Vec::new();
    for root in 0..n {
        if !remaining[root] {
            continue;
        }
        let members = (0..n)
            .filter(|node| mutual[root * n + node] != 0)
            .collect::<Vec<_>>();
        for &node in &members {
            remaining[node] = false;
        }
        classes.push(members);
    }
    classes
}

fn dag_depth(dag: &[u8], nodes: usize) -> Vec<usize> {
    let mut depth = vec![0usize; nodes];
    let mut pending = vec![true; nodes];
    let mut remaining = nodes;
    while remaining != 0 {
        let mut ready = Vec::new();
        for node in 0..nodes {
            if pending[node]
                && !(0..nodes).any(|parent| pending[parent] && dag[parent * nodes + node] != 0)
            {
                ready.push(node);
            }
        }
        if ready.is_empty() {
            // A valid SCC condensation is a DAG; return a sentinel only if the
            // input or port violated that invariant so packet validation fails.
            return vec![usize::MAX; nodes];
        }
        for node in ready {
            for child in 0..nodes {
                if dag[node * nodes + child] != 0 {
                    depth[child] = depth[child].max(depth[node].saturating_add(1));
                }
            }
            pending[node] = false;
            remaining -= 1;
        }
    }
    depth
}
