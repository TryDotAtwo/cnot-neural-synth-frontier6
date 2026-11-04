//! Ordered-pair diffusion class profiles.

use super::super::super::DiffusionConfig;
use super::super::chain::DiffusionChain;
use std::collections::HashMap;

pub(in crate::diffusion) struct PairProfile {
    pub(in crate::diffusion) states: Vec<[usize; 4]>,
    class_lookup: HashMap<[usize; 4], usize>,
    pub(in crate::diffusion) log_class_size: Vec<f64>,
    pub(in crate::diffusion) stationary: Vec<f64>,
    pub(in crate::diffusion) masses: Vec<f64>,
    pub(in crate::diffusion) exact_zero: Vec<bool>,
}

impl PairProfile {
    pub(in crate::diffusion) fn build(n: usize, config: DiffusionConfig) -> Option<Self> {
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

    pub(in crate::diffusion) fn class_index(&self, histogram: &[usize; 4]) -> Option<usize> {
        self.class_lookup.get(histogram).copied()
    }

    pub(in crate::diffusion) fn raw(&self, time: usize, class: usize) -> f64 {
        self.masses[time * self.states.len() + class]
    }
}

pub(in crate::diffusion) fn pair_state_count(n: usize) -> Option<usize> {
    let top = n.checked_add(3)?;
    let product = top
        .checked_mul(top.checked_sub(1)?)?
        .checked_mul(top.checked_sub(2)?)?;
    product
        .checked_div(6)?
        .checked_sub(1)?
        .checked_sub(n.checked_mul(3)?)
}
