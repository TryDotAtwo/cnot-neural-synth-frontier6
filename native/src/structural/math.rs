//! Packed GF(2), dense matrix, and graph kernels shared by structural features.

pub(super) fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("feature allocation failed: {error}"))?;
    values.resize(len, T::default());
    Ok(values)
}

pub(super) fn gf2_rank(rows: &[u64], n: usize) -> usize {
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

pub(super) fn packed_to_dense(rows: &[u64], n: usize) -> Vec<i64> {
    let words = n.div_ceil(64);
    (0..n * n)
        .map(|index| {
            ((rows[(index / n) * words + (index % n) / 64] >> ((index % n) % 64)) & 1) as i64
        })
        .collect()
}

pub(super) fn transpose_dense(matrix: &[i64], n: usize) -> Vec<i64> {
    let mut result = vec![0; matrix.len()];
    for i in 0..n {
        for j in 0..n {
            result[j * n + i] = matrix[i * n + j];
        }
    }
    result
}

pub(super) fn row_sums(matrix: &[i64], n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| matrix[i * n..(i + 1) * n].iter().sum())
        .collect()
}

pub(super) fn col_sums(matrix: &[i64], n: usize) -> Vec<i64> {
    (0..n)
        .map(|j| (0..n).map(|i| matrix[i * n + j]).sum())
        .collect()
}

pub(super) fn matmul(left: &[i64], right: &[i64], n: usize) -> Result<Vec<i64>, String> {
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

pub(super) fn hadamard(left: &[i64], right: &[i64], n: usize) -> Vec<i64> {
    (0..n * n).map(|i| left[i] * right[i]).collect()
}

pub(super) fn undirected_adjacency(adjacency: &[i64], n: usize) -> Vec<i64> {
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

pub(super) fn undirected_components(adjacency: &[i64], n: usize) -> Vec<Vec<usize>> {
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

pub(super) fn transitive_closure(adjacency: &[i64], n: usize) -> Vec<bool> {
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

pub(super) fn equivalence_classes(mutual: &[u8], n: usize) -> Vec<Vec<usize>> {
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

pub(super) fn dag_depth(dag: &[u8], nodes: usize) -> Vec<usize> {
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
