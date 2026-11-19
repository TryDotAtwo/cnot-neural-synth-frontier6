//! Stage18 bipartite graph matrices. The graph distance is 2|V| + 2|E|
//! + minimum vertex cover. A disjoint matching lower-bounds every cover.
//! A verified reduction reaching this lower bound proves the claimed distance.
use crate::stage35::bit;
use crate::MatrixState;
use std::collections::{HashSet, VecDeque};

pub(super) fn lower_bound(state: &MatrixState) -> Option<u64> {
    let n = state.dimension() as usize;
    if (0..n).any(|i| !bit(state, i, i)) {
        return None;
    }
    // Identify all off-diagonal supports. Every physical row must be one
    // of z, q_v, x_v, y_e with its exact prescribed support.
    let off: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i && bit(state, i, j)).collect())
        .collect();
    let mut roots = (0..n).filter(|&i| off[i].is_empty());
    let z = roots.next()?;
    if roots.next().is_some() {
        return None;
    }
    let q: Vec<_> = (0..n).filter(|&i| off[i] == [z]).collect();
    if q.is_empty() {
        return None;
    }
    let mut vertex_at = vec![None; n];
    let mut used = vec![false; n];
    used[z] = true;
    for (v, &qi) in q.iter().enumerate() {
        let mut candidates = (0..n).filter(|&i| off[i] == [qi]);
        let x = candidates.next()?;
        if candidates.next().is_some() || used[x] || used[qi] || x == qi {
            return None;
        }
        used[qi] = true;
        used[x] = true;
        vertex_at[x] = Some(v);
    }
    let mut edges = HashSet::new();
    let mut adj = vec![Vec::new(); q.len()];
    for i in (0..n).filter(|&i| !used[i]) {
        if off[i].len() != 3 || !off[i].contains(&z) {
            return None;
        }
        let mut endpoints = off[i].iter().copied().filter(|&j| j != z);
        let a = vertex_at[endpoints.next()?]?;
        let b = vertex_at[endpoints.next()?]?;
        if !edges.insert((a.min(b), a.max(b))) {
            return None;
        }
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut color = vec![None; q.len()];
    for root in 0..q.len() {
        if color[root].is_some() {
            continue;
        }
        color[root] = Some(false);
        let mut queue = VecDeque::from([root]);
        while let Some(a) = queue.pop_front() {
            let other = !color[a]?;
            for &b in &adj[a] {
                if let Some(visited) = color[b] {
                    if visited != other {
                        return None;
                    }
                } else {
                    color[b] = Some(other);
                    queue.push_back(b);
                }
            }
        }
    }
    let mut matching = vec![None; q.len()];
    for a in (0..q.len()).filter(|&a| color[a] == Some(false)) {
        augment(a, &adj, &mut matching, &mut vec![false; q.len()]);
    }
    let size = matching.iter().flatten().count();
    Some(2 * q.len() as u64 + 2 * edges.len() as u64 + size as u64)
}

fn augment(
    a: usize,
    adj: &[Vec<usize>],
    matching: &mut [Option<usize>],
    seen: &mut [bool],
) -> bool {
    for &b in &adj[a] {
        if seen[b] {
            continue;
        }
        seen[b] = true;
        if matching[b].is_none_or(|old| augment(old, adj, matching, seen)) {
            matching[b] = Some(a);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(vertices: usize, edges: &[(usize, usize)]) -> MatrixState {
        let n = 1 + 2 * vertices + edges.len();
        let mut rows: Vec<_> = (0..n).map(|i| 1u64 << i).collect();
        for v in 0..vertices {
            rows[1 + v] |= 1;
            rows[1 + vertices + v] |= 1 << (1 + v);
        }
        for (e, &(a, b)) in edges.iter().enumerate() {
            rows[1 + 2 * vertices + e] |= 1 | (1 << (1 + vertices + a)) | (1 << (1 + vertices + b));
        }
        MatrixState::from_packed_rows(n as u32, "graph-test", rows).unwrap()
    }

    #[test]
    fn validates_every_register_and_bipartite_matching() {
        let state = matrix(4, &[(0, 2), (0, 3), (1, 2)]);
        assert_eq!(lower_bound(&state), Some(16));
        assert_eq!(lower_bound(&matrix(3, &[(0, 1), (1, 2), (2, 0)])), None);
        assert_eq!(lower_bound(&matrix(2, &[(0, 1), (1, 0)])), None);
        let mut rows = state.packed_rows().to_vec();
        *rows.last_mut().unwrap() ^= 1;
        let malformed =
            MatrixState::from_packed_rows(state.dimension(), "graph-test", rows).unwrap();
        assert_eq!(lower_bound(&malformed), None);
    }
}
