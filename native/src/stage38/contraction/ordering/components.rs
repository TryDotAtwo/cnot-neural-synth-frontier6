//! Connected-component partitioning for the active factor graph.

use super::super::factorization::Factor;
use std::collections::BTreeSet;

pub(in crate::stage38::contraction) fn components(
    event_count: usize,
    factors: &[Factor],
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![BTreeSet::new(); event_count];
    for factor in factors {
        for variable in factor.scope.iter().copied() {
            adjacency[variable].extend(
                factor
                    .scope
                    .iter()
                    .copied()
                    .filter(|other| *other != variable),
            );
        }
    }
    let mut seen = vec![false; event_count];
    let mut output = Vec::new();
    for start in 0..event_count {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(variable) = stack.pop() {
            component.push(variable);
            for neighbor in adjacency[variable].iter().copied() {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        output.push(component);
    }
    output
}
