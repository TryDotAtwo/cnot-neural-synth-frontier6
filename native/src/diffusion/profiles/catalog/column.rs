//! Single-column diffusion class profiles.

use super::super::super::DiffusionConfig;
use super::super::chain::{binomial_f64, DiffusionChain};

#[derive(Clone, Copy)]
pub(in crate::diffusion) struct DiffusionSample {
    pub(in crate::diffusion) raw: f64,
    pub(in crate::diffusion) stationary: f64,
    pub(in crate::diffusion) log_mass: f64,
    pub(in crate::diffusion) exact_zero: bool,
    pub(in crate::diffusion) underflow: bool,
}

pub(in crate::diffusion) struct ColumnProfile {
    n: usize,
    pub(in crate::diffusion) states: Vec<(usize, usize)>,
    class_lookup: Vec<usize>,
    pub(in crate::diffusion) log_class_size: Vec<f64>,
    pub(in crate::diffusion) stationary: Vec<f64>,
    masses: Vec<f64>,
    pub(in crate::diffusion) exact_zero: Vec<bool>,
    config: DiffusionConfig,
}

impl ColumnProfile {
    pub(in crate::diffusion) fn build(n: usize, config: DiffusionConfig) -> Option<Self> {
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

    pub(in crate::diffusion) fn class_index(
        &self,
        bit: usize,
        other_weight: usize,
    ) -> Option<usize> {
        let index = *self
            .class_lookup
            .get(bit.checked_mul(self.n)?.checked_add(other_weight)?)?;
        (index != usize::MAX).then_some(index)
    }

    pub(in crate::diffusion) fn raw(&self, time: usize, class: usize) -> f64 {
        self.masses[time * self.states.len() + class]
    }

    pub(in crate::diffusion) fn sample(&self, time: usize, class: usize) -> DiffusionSample {
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
