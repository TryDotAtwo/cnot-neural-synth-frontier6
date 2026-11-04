//! Deterministic min-fill elimination order and induced-bag accounting.

use super::super::super::Stage38Error;
use super::super::factorization::Factor;
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub(in crate::stage38::contraction) fn elimination_order(
    vertices: &[usize],
    factors: &[Factor],
    protected: &HashSet<usize>,
) -> Result<(Vec<usize>, usize), Stage38Error> {
    let mut adjacency: BTreeMap<usize, BTreeSet<usize>> = vertices
        .iter()
        .copied()
        .map(|v| (v, BTreeSet::new()))
        .collect();
    for factor in factors {
        for variable in factor.scope.iter().copied() {
            for other in factor
                .scope
                .iter()
                .copied()
                .filter(|other| *other != variable)
            {
                adjacency
                    .get_mut(&variable)
                    .ok_or(Stage38Error::ContractionInvariant(
                        "factor scope is outside component",
                    ))?
                    .insert(other);
            }
        }
    }

    let remaining: BTreeSet<usize> = vertices
        .iter()
        .copied()
        .filter(|vertex| !protected.contains(vertex))
        .collect();
    let mut seen = HashSet::new();
    let mut pseudoforest = true;
    for start in remaining.iter().copied() {
        if seen.contains(&start) {
            continue;
        }
        seen.insert(start);
        let mut stack = vec![start];
        let mut vertex_count = 0usize;
        let mut degree_sum = 0usize;
        while let Some(vertex) = stack.pop() {
            vertex_count += 1;
            let neighbors: Vec<usize> = adjacency[&vertex]
                .intersection(&remaining)
                .copied()
                .collect();
            degree_sum = degree_sum
                .checked_add(neighbors.len())
                .ok_or(Stage38Error::SizeOverflow)?;
            for neighbor in neighbors {
                if seen.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        if degree_sum / 2 > vertex_count {
            pseudoforest = false;
        }
    }
    let use_protected = !protected.is_empty() && pseudoforest;
    let mut order = Vec::with_capacity(vertices.len());
    let mut max_bag = 0usize;

    while !adjacency.is_empty() {
        let pool: Vec<usize> = if use_protected {
            adjacency
                .keys()
                .copied()
                .filter(|vertex| !protected.contains(vertex))
                .collect()
        } else {
            Vec::new()
        };
        let selected = if !pool.is_empty() {
            let vertex = pool
                .into_iter()
                .min_by_key(|vertex| {
                    (
                        adjacency[vertex]
                            .iter()
                            .filter(|neighbor| !protected.contains(neighbor))
                            .count(),
                        elimination_key(*vertex, &adjacency),
                    )
                })
                .ok_or(Stage38Error::ContractionInvariant("empty elimination pool"))?;
            if adjacency[&vertex]
                .iter()
                .filter(|neighbor| !protected.contains(neighbor))
                .count()
                > 2
            {
                return Err(Stage38Error::ContractionInvariant(
                    "pseudoforest elimination degree exceeded two",
                ));
            }
            vertex
        } else {
            adjacency
                .keys()
                .copied()
                .min_by_key(|vertex| elimination_key(*vertex, &adjacency))
                .ok_or(Stage38Error::ContractionInvariant(
                    "no elimination candidate",
                ))?
        };

        let neighbors: Vec<usize> = adjacency[&selected].iter().copied().collect();
        max_bag = max_bag.max(
            neighbors
                .len()
                .checked_add(1)
                .ok_or(Stage38Error::SizeOverflow)?,
        );
        for (position, left) in neighbors.iter().copied().enumerate() {
            for right in neighbors.iter().copied().skip(position + 1) {
                adjacency.get_mut(&left).unwrap().insert(right);
                adjacency.get_mut(&right).unwrap().insert(left);
            }
        }
        for neighbor in &neighbors {
            adjacency.get_mut(neighbor).unwrap().remove(&selected);
        }
        adjacency.remove(&selected);
        order.push(selected);
    }
    Ok((order, max_bag))
}

fn elimination_key(
    vertex: usize,
    adjacency: &BTreeMap<usize, BTreeSet<usize>>,
) -> (usize, usize, usize) {
    let neighbors = &adjacency[&vertex];
    let mut fill = 0usize;
    for (position, left) in neighbors.iter().copied().enumerate() {
        for right in neighbors.iter().copied().skip(position + 1) {
            if !adjacency[&left].contains(&right) {
                fill += 1;
            }
        }
    }
    (fill, neighbors.len(), vertex)
}
