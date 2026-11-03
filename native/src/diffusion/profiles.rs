//! Exact finite-state profiles and transition chains for source-relative diffusion features.

use super::{DiffusionConfig, MAX_UNIFORMIZATION_TERMS, POISSON_TAIL_MARGIN, POISSON_TAIL_SIGMAS};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(super) struct DiffusionSample {
    pub(super) raw: f64,
    pub(super) stationary: f64,
    pub(super) log_mass: f64,
    pub(super) exact_zero: bool,
    pub(super) underflow: bool,
}

pub(super) struct ColumnProfile {
    n: usize,
    pub(super) states: Vec<(usize, usize)>,
    class_lookup: Vec<usize>,
    pub(super) log_class_size: Vec<f64>,
    pub(super) stationary: Vec<f64>,
    masses: Vec<f64>,
    pub(super) exact_zero: Vec<bool>,
    config: DiffusionConfig,
}

impl ColumnProfile {
    pub(super) fn build(n: usize, config: DiffusionConfig) -> Option<Self> {
        if n == 0 || n > config.max_n as usize {
            return None;
        }
        let state_count = n.checked_mul(2)?.checked_sub(1)?;
        let lookup_len = n.checked_mul(2)?;
        let mut states = Vec::new();
        states.try_reserve_exact(state_count).ok()?;
        let mut class_lookup = vec![usize::MAX; lookup_len];
        for bit in 0..=1usize {
            for other_weight in 0..n {
                if bit + other_weight == 0 {
                    continue;
                }
                let index = states.len();
                states.push((bit, other_weight));
                class_lookup[bit * n + other_weight] = index;
            }
        }
        if states.len() != state_count {
            return None;
        }

        let mut class_sizes = Vec::new();
        class_sizes.try_reserve_exact(state_count).ok()?;
        for &(bit, other_weight) in &states {
            let _ = bit;
            class_sizes.push(binomial_f64(n - 1, other_weight)?);
        }
        let total_classes: f64 = class_sizes.iter().sum();
        if !total_classes.is_finite() || total_classes <= 0.0 {
            return None;
        }
        let log_class_size = class_sizes.iter().map(|size| size.ln()).collect();
        let stationary = class_sizes
            .iter()
            .map(|size| size / total_classes)
            .collect::<Vec<_>>();
        let chain = DiffusionChain::build_columns(n, &states, &class_lookup)?;
        let initial = *class_lookup.get(n)?;
        if initial == usize::MAX {
            return None;
        }

        let mass_count = state_count.checked_mul(4)?;
        let mut masses = Vec::new();
        masses.try_reserve_exact(mass_count).ok()?;
        let mut exact_zero = Vec::new();
        exact_zero.try_reserve_exact(mass_count).ok()?;
        for tau in config.taus {
            let time = n as f64 * tau;
            if !time.is_finite() {
                return None;
            }
            let distribution = chain.distribution(initial, time)?;
            masses.extend(distribution.iter().copied());
            exact_zero.extend((0..state_count).map(|index| time == 0.0 && index != initial));
        }

        Some(Self {
            n,
            states,
            class_lookup,
            log_class_size,
            stationary,
            masses,
            exact_zero,
            config,
        })
    }

    pub(super) fn class_index(&self, bit: usize, other_weight: usize) -> Option<usize> {
        let index = *self
            .class_lookup
            .get(bit.checked_mul(self.n)?.checked_add(other_weight)?)?;
        (index != usize::MAX).then_some(index)
    }

    pub(super) fn raw(&self, time: usize, class: usize) -> f64 {
        self.masses[time * self.states.len() + class]
    }

    pub(super) fn sample(&self, time: usize, class: usize) -> DiffusionSample {
        let raw = self.raw(time, class);
        let stationary = self.stationary[class];
        let exact_zero = self.exact_zero[time * self.states.len() + class];
        let underflow = (raw == 0.0 && !exact_zero)
            || (raw > 0.0 && (raw as f32) == 0.0)
            || (stationary > 0.0 && (stationary as f32) == 0.0);
        DiffusionSample {
            raw,
            stationary,
            log_mass: raw.max(self.config.floor).ln(),
            exact_zero,
            underflow,
        }
    }
}

pub(super) struct PairProfile {
    pub(super) states: Vec<[usize; 4]>,
    class_lookup: HashMap<[usize; 4], usize>,
    pub(super) log_class_size: Vec<f64>,
    pub(super) stationary: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) exact_zero: Vec<bool>,
}

impl PairProfile {
    pub(super) fn build(n: usize, config: DiffusionConfig) -> Option<Self> {
        if n < 2 {
            return None;
        }
        let state_count = pair_state_count(n)?;
        if state_count > config.max_pair_states {
            return None;
        }

        let mut states = Vec::new();
        states.try_reserve_exact(state_count).ok()?;
        for h0 in 0..=n {
            for h1 in 0..=n - h0 {
                for h2 in 0..=n - h0 - h1 {
                    let h3 = n - h0 - h1 - h2;
                    if usize::from(h1 > 0) + usize::from(h2 > 0) + usize::from(h3 > 0) >= 2 {
                        states.push([h0, h1, h2, h3]);
                    }
                }
            }
        }
        if states.len() != state_count {
            return None;
        }
        let mut class_lookup = HashMap::new();
        class_lookup.try_reserve(state_count).ok()?;
        class_lookup.extend(
            states
                .iter()
                .copied()
                .enumerate()
                .map(|(index, state)| (state, index)),
        );

        let mut log_factorial = Vec::new();
        log_factorial.try_reserve_exact(n.checked_add(1)?).ok()?;
        log_factorial.push(0.0);
        for value in 1..=n {
            let previous = *log_factorial.last()?;
            log_factorial.push(previous + (value as f64).ln());
        }
        let log_class_size = states
            .iter()
            .map(|histogram| {
                log_factorial[n]
                    - histogram
                        .iter()
                        .map(|count| log_factorial[*count])
                        .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let max_log_size = log_class_size
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let scaled_total: f64 = log_class_size
            .iter()
            .map(|value| (value - max_log_size).exp())
            .sum();
        if !scaled_total.is_finite() || scaled_total <= 0.0 {
            return None;
        }
        let log_total = max_log_size + scaled_total.ln();
        let stationary = log_class_size
            .iter()
            .map(|value| (value - log_total).exp())
            .collect::<Vec<_>>();

        let chain = DiffusionChain::build_pairs(n, &states, &class_lookup)?;
        let initial_state = [n.checked_sub(2)?, 1, 1, 0];
        let initial = *class_lookup.get(&initial_state)?;
        let mass_count = state_count.checked_mul(4)?;
        let mut masses = Vec::new();
        masses.try_reserve_exact(mass_count).ok()?;
        let mut exact_zero = Vec::new();
        exact_zero.try_reserve_exact(mass_count).ok()?;
        for tau in config.taus {
            let time = n as f64 * tau;
            if !time.is_finite() {
                return None;
            }
            let distribution = chain.distribution(initial, time)?;
            masses.extend(distribution.iter().copied());
            exact_zero.extend((0..state_count).map(|index| time == 0.0 && index != initial));
        }
        Some(Self {
            states,
            class_lookup,
            log_class_size,
            stationary,
            masses,
            exact_zero,
        })
    }

    pub(super) fn class_index(&self, histogram: &[usize; 4]) -> Option<usize> {
        self.class_lookup.get(histogram).copied()
    }

    pub(super) fn raw(&self, time: usize, class: usize) -> f64 {
        self.masses[time * self.states.len() + class]
    }
}

pub(super) fn pair_state_count(n: usize) -> Option<usize> {
    let top = n.checked_add(3)?;
    let product = top
        .checked_mul(top.checked_sub(1)?)?
        .checked_mul(top.checked_sub(2)?)?;
    product
        .checked_div(6)?
        .checked_sub(1)?
        .checked_sub(n.checked_mul(3)?)
}

struct DiffusionChain {
    transitions: Vec<Vec<(usize, f64)>>,
}

impl DiffusionChain {
    fn build_columns(n: usize, states: &[(usize, usize)], lookup: &[usize]) -> Option<Self> {
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

    fn build_pairs(
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

    fn distribution(&self, initial: usize, time: f64) -> Option<Vec<f64>> {
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

fn binomial_f64(n: usize, k: usize) -> Option<f64> {
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
