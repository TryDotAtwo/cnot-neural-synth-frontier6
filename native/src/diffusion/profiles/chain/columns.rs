//! Transition construction for one-column weight states.

pub(super) fn build_columns(
    n: usize,
    states: &[(usize, usize)],
    lookup: &[usize],
) -> Option<Vec<Vec<(usize, f64)>>> {
    let rate = n.checked_mul(n.checked_sub(1)?)?;
    let mut transitions = Vec::new();
    transitions.try_reserve_exact(states.len()).ok()?;
    for (index, &(bit, other_weight)) in states.iter().enumerate() {
        if rate == 0 {
            transitions.push(vec![(index, 1.0)]);
            continue;
        }
        let mut edges = Vec::<(usize, usize)>::with_capacity(4);
        let self_rate = n
            .checked_sub(bit.checked_add(other_weight)?)?
            .checked_mul(n - 1)?;
        push_edge(&mut edges, index, self_rate);
        if other_weight > 0 {
            let destination = *lookup.get((1 - bit).checked_mul(n)?.checked_add(other_weight)?)?;
            push_edge(&mut edges, destination, other_weight);
        }
        if other_weight + 1 < n {
            let destination = *lookup.get(bit.checked_mul(n)?.checked_add(other_weight + 1)?)?;
            let up_rate = bit
                .checked_add(other_weight)?
                .checked_mul(n - 1 - other_weight)?;
            push_edge(&mut edges, destination, up_rate);
        }
        if other_weight > 0 {
            let destination = *lookup.get(bit.checked_mul(n)?.checked_add(other_weight - 1)?)?;
            let down_rate =
                other_weight.checked_mul(bit.checked_add(other_weight)?.checked_sub(1)?)?;
            push_edge(&mut edges, destination, down_rate);
        }
        if edges
            .iter()
            .try_fold(0usize, |sum, (_, count)| sum.checked_add(*count))?
            != rate
        {
            return None;
        }
        transitions.push(
            edges
                .into_iter()
                .filter(|(_, count)| *count > 0)
                .map(|(destination, count)| (destination, count as f64 / rate as f64))
                .collect(),
        );
    }
    Some(transitions)
}

fn push_edge(edges: &mut Vec<(usize, usize)>, destination: usize, rate: usize) {
    if rate > 0 {
        edges.push((destination, rate));
    }
}
