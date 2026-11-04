//! Transition construction for joint two-column histogram states.

use std::collections::HashMap;

pub(super) fn build_pairs(
    n: usize,
    states: &[[usize; 4]],
    lookup: &HashMap<[usize; 4], usize>,
) -> Option<Vec<Vec<(usize, f64)>>> {
    let rate = n.checked_mul(n.checked_sub(1)?)?;
    if rate == 0 {
        return None;
    }
    let mut transitions = Vec::new();
    transitions.try_reserve_exact(states.len()).ok()?;
    for histogram in states {
        let mut edges = Vec::<(usize, usize)>::with_capacity(16);
        let mut outgoing = 0usize;
        for control_pattern in 0..4 {
            for target_pattern in 0..4 {
                if histogram[control_pattern] == 0
                    || (control_pattern == target_pattern && histogram[target_pattern] < 2)
                {
                    continue;
                }
                let target_count =
                    histogram[target_pattern] - usize::from(control_pattern == target_pattern);
                let ways = histogram[control_pattern].checked_mul(target_count)?;
                if ways == 0 {
                    continue;
                }
                let mut successor = *histogram;
                successor[target_pattern] = successor[target_pattern].checked_sub(1)?;
                let destination_pattern = control_pattern ^ target_pattern;
                successor[destination_pattern] = successor[destination_pattern].checked_add(1)?;
                let destination = *lookup.get(&successor)?;
                edges.push((destination, ways));
                outgoing = outgoing.checked_add(ways)?;
            }
        }
        if outgoing != rate {
            return None;
        }
        transitions.push(
            edges
                .into_iter()
                .map(|(destination, ways)| (destination, ways as f64 / rate as f64))
                .collect(),
        );
    }
    Some(transitions)
}
