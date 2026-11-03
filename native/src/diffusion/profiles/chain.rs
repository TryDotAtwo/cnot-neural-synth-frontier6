use super::super::{MAX_UNIFORMIZATION_TERMS, POISSON_TAIL_MARGIN, POISSON_TAIL_SIGMAS};
use std::collections::HashMap;

pub(super) struct DiffusionChain {
    transitions: Vec<Vec<(usize, f64)>>,
}

impl DiffusionChain {
    pub(super) fn build_columns(
        n: usize,
        states: &[(usize, usize)],
        lookup: &[usize],
    ) -> Option<Self> {
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
                let destination =
                    *lookup.get((1 - bit).checked_mul(n)?.checked_add(other_weight)?)?;
                push_edge(&mut edges, destination, other_weight);
            }
            if other_weight + 1 < n {
                let destination =
                    *lookup.get(bit.checked_mul(n)?.checked_add(other_weight + 1)?)?;
                let up_rate = bit
                    .checked_add(other_weight)?
                    .checked_mul(n - 1 - other_weight)?;
                push_edge(&mut edges, destination, up_rate);
            }
            if other_weight > 0 {
                let destination =
                    *lookup.get(bit.checked_mul(n)?.checked_add(other_weight - 1)?)?;
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
        Some(Self { transitions })
    }

    pub(super) fn build_pairs(
        n: usize,
        states: &[[usize; 4]],
        lookup: &HashMap<[usize; 4], usize>,
    ) -> Option<Self> {
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
                    successor[destination_pattern] =
                        successor[destination_pattern].checked_add(1)?;
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
        Some(Self { transitions })
    }

    pub(super) fn distribution(&self, initial: usize, time: f64) -> Option<Vec<f64>> {
        if time == 0.0 {
            let mut mass = vec![0.0; self.transitions.len()];
            *mass.get_mut(initial)? = 1.0;
            return Some(mass);
        }
        let weights = normalized_poisson_weights(time)?;
        let mut current = vec![0.0; self.transitions.len()];
        *current.get_mut(initial)? = 1.0;
        let mut mass = vec![0.0; self.transitions.len()];
        for (step, weight) in weights.iter().copied().enumerate() {
            if weight != 0.0 {
                for (total, value) in mass.iter_mut().zip(&current) {
                    *total += weight * value;
                }
            }
            if step + 1 < weights.len() {
                let mut next = vec![0.0; self.transitions.len()];
                for (source, value) in current.iter().copied().enumerate() {
                    if value == 0.0 {
                        continue;
                    }
                    for &(destination, probability) in &self.transitions[source] {
                        *next.get_mut(destination)? += value * probability;
                    }
                }
                current = next;
            }
        }
        let total: f64 = mass.iter().sum();
        if !total.is_finite() || total <= 0.0 || mass.iter().any(|value| !value.is_finite()) {
            return None;
        }
        mass.iter_mut().for_each(|value| *value /= total);
        Some(mass)
    }
}

fn push_edge(edges: &mut Vec<(usize, usize)>, destination: usize, rate: usize) {
    if rate > 0 {
        edges.push((destination, rate));
    }
}

fn normalized_poisson_weights(time: f64) -> Option<Vec<f64>> {
    if !time.is_finite() || time <= 0.0 {
        return None;
    }
    let margin = POISSON_TAIL_SIGMAS * time.sqrt() + POISSON_TAIL_MARGIN;
    let upper_float = (time + margin).ceil();
    if !upper_float.is_finite()
        || upper_float < 0.0
        || upper_float > MAX_UNIFORMIZATION_TERMS as f64
    {
        return None;
    }
    let upper = upper_float as usize;
    let lower = (time - margin).floor().max(0.0) as usize;
    let mode = time.floor() as usize;
    let mut weights = Vec::new();
    weights.try_reserve_exact(upper.checked_add(1)?).ok()?;
    weights.resize(upper + 1, 0.0);
    *weights.get_mut(mode)? = 1.0;
    let mut index = mode;
    while index > lower {
        let previous = weights[index] * index as f64 / time;
        weights[index - 1] = previous;
        index -= 1;
    }
    for index in mode..upper {
        weights[index + 1] = weights[index] * time / (index + 1) as f64;
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    weights.iter_mut().for_each(|weight| *weight /= total);
    Some(weights)
}

pub(super) fn binomial_f64(n: usize, k: usize) -> Option<f64> {
    if k > n {
        return None;
    }
    let k = k.min(n - k);
    let mut value = 1.0f64;
    for index in 1..=k {
        value *= (n - k + index) as f64 / index as f64;
    }
    (value.is_finite() && value > 0.0).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::normalized_poisson_weights;

    #[test]
    fn centered_poisson_weights_remain_normalized_at_large_time() {
        for time in [1e-12, 0.5, 64.0, 4096.0] {
            let weights = normalized_poisson_weights(time).unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(weights
                .iter()
                .all(|weight| *weight >= 0.0 && weight.is_finite()));
        }
    }
}
