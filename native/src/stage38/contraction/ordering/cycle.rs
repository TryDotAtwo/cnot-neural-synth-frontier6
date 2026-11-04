//! Detect mandatory precedence cycles in factorized history constraints.

use super::super::super::Constraint;
use super::super::factorization::Factor;
use std::collections::{BTreeSet, HashSet};

pub(in crate::stage38::contraction) fn has_mandatory_cycle(
    event_count: usize,
    factors: &[Factor],
) -> bool {
    let mut edges = vec![BTreeSet::new(); event_count];
    for factor in factors {
        let Some(first) = factor.terms.first() else {
            continue;
        };
        let mut mandatory: BTreeSet<Constraint> = first.iter().copied().collect();
        for term in factor.terms.iter().skip(1) {
            let constraints: HashSet<Constraint> = term.iter().copied().collect();
            mandatory.retain(|constraint| constraints.contains(constraint));
        }
        for (before, after) in mandatory {
            edges[before].insert(after);
        }
    }
    let mut indegree = vec![0usize; event_count];
    for targets in &edges {
        for target in targets {
            indegree[*target] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..event_count)
        .filter(|event| indegree[*event] == 0)
        .collect();
    let mut visited = 0usize;
    while let Some(event) = ready.pop() {
        visited += 1;
        for target in edges[event].iter().copied() {
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(target);
            }
        }
    }
    visited != event_count
}
