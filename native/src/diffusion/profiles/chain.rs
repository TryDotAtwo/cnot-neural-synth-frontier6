//! Finite-state diffusion transition builders and uniformized propagation.

mod columns;
mod distribution;
mod pairs;

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
        Some(Self {
            transitions: columns::build_columns(n, states, lookup)?,
        })
    }

    pub(super) fn build_pairs(
        n: usize,
        states: &[[usize; 4]],
        lookup: &HashMap<[usize; 4], usize>,
    ) -> Option<Self> {
        Some(Self {
            transitions: pairs::build_pairs(n, states, lookup)?,
        })
    }

    pub(super) fn distribution(&self, initial: usize, time: f64) -> Option<Vec<f64>> {
        distribution::distribution(&self.transitions, initial, time)
    }
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
    use super::distribution::normalized_poisson_weights;

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
