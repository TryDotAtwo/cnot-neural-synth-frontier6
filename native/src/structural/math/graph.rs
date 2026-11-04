//! Directed and undirected reachability and component kernels.

pub(in crate::structural) fn undirected_adjacency(adjacency: &[i64], n: usize) -> Vec<i64> {
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

pub(in crate::structural) fn undirected_components(adjacency: &[i64], n: usize) -> Vec<Vec<usize>> {
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

pub(in crate::structural) fn transitive_closure(adjacency: &[i64], n: usize) -> Vec<bool> {
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

pub(in crate::structural) fn equivalence_classes(mutual: &[u8], n: usize) -> Vec<Vec<usize>> {
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

pub(in crate::structural) fn dag_depth(dag: &[u8], nodes: usize) -> Vec<usize> {
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
