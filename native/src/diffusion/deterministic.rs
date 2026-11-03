//! Exact finite-state diffusion feature producer for G01-G04 and G06.

use super::profiles::pair_state_count;
use super::{ProfileCache, unavailable};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureValue};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;
use std::sync::Arc;

pub(super) struct DiffusionProducer {
    spec: FeatureSpec,
    cache: Arc<ProfileCache>,
}

impl FeatureProducer for DiffusionProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        if self.spec.id.ordinal() == 6 {
            return self.compute_pair(state, n);
        }
        if n > self.cache.config.max_n as usize {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "column diffusion max_n={} exceeded by n={n}",
                    self.cache.config.max_n
                ),
            );
        }
        let Some(profile) = self.cache.get(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion checked state/iteration budget exceeded".into(),
            );
        };
        let Some(value_count) = n.checked_mul(self.spec.channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion output allocation failed".into(),
            );
        }
        values.resize(value_count, 0.0f32);

        let row_weights = row_weights(state);
        let column_weights = column_weights(state);
        for wire in 0..n {
            let diagonal_bit = matrix_bit(state, wire, wire) as usize;
            let weight = if self.spec.id.ordinal() == 2 {
                row_weights[wire]
            } else {
                column_weights[wire]
            };
            let Some(other_weight) = weight.checked_sub(diagonal_bit) else {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "diagonal bit exceeds selected row/column weight".into(),
                );
            };
            let Some(class) = profile.class_index(diagonal_bit, other_weight) else {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "selected row/column class is outside the diffusion chain".into(),
                );
            };
            let base = wire * self.spec.channels;
            match self.spec.id.ordinal() {
                1 | 2 => {
                    for time_index in 0..4 {
                        let sample = profile.sample(time_index, class);
                        let offset = base + time_index * 7;
                        values[offset] = sample.raw as f32;
                        values[offset + 1] = profile.log_class_size[class] as f32;
                        values[offset + 2] = sample.stationary as f32;
                        values[offset + 3] = (sample.log_mass - sample.stationary.ln()) as f32;
                        values[offset + 4] = f32::from(sample.exact_zero);
                        values[offset + 5] = f32::from(sample.underflow);
                        values[offset + 6] = f32::from(sample.raw < self.cache.config.floor);
                    }
                }
                3 => {
                    for time_index in 0..4 {
                        let sample = profile.sample(time_index, class);
                        let offset = base + time_index * 4;
                        values[offset] = sample.log_mass as f32;
                        values[offset + 1] = f32::from(sample.exact_zero);
                        values[offset + 2] = f32::from(sample.underflow);
                        values[offset + 3] = f32::from(sample.raw < self.cache.config.floor);
                    }
                }
                4 => {
                    for time_index in 0..3 {
                        let current = profile.raw(time_index + 1, class);
                        let previous = profile.raw(time_index, class);
                        values[base + time_index] = (current - previous) as f32;
                    }
                }
                _ => {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "diffusion producer received an unregistered feature ID".into(),
                    );
                }
            }
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, self.spec.channels],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl DiffusionProducer {
    fn compute_pair(&self, state: &MatrixState, n: usize) -> FeatureComputation {
        if n < 2 {
            return unavailable(
                FeatureStatus::Unsupported,
                "rank-2 pair projection requires n>=2".into(),
            );
        }
        let Some(state_count) = pair_state_count(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion state count overflow".into(),
            );
        };
        if state_count > self.cache.config.max_pair_states {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "pair diffusion resource_limit: {state_count} states exceeds {}",
                    self.cache.config.max_pair_states
                ),
            );
        }
        let Some(profile) = self.cache.get_pair(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion checked allocation/iteration budget exceeded".into(),
            );
        };
        let Some(pair_count) = n.checked_mul(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output size overflow".into(),
            );
        };
        let Some(value_count) = pair_count.checked_mul(self.spec.channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output allocation failed".into(),
            );
        }
        values.resize(value_count, 0.0f32);

        for first in 0..n {
            for second in 0..n {
                if first == second {
                    continue;
                }
                let histogram = pair_histogram(state, first, second);
                let Some(class) = profile.class_index(&histogram) else {
                    return unavailable(
                        FeatureStatus::Inapplicable,
                        "matrix column pair is outside the rank-2 diffusion chain".into(),
                    );
                };
                let base = (first * n + second) * self.spec.channels;
                for time_index in 0..4 {
                    let raw = profile.raw(time_index, class);
                    let stationary = profile.stationary[class];
                    let exact_zero = profile.exact_zero[time_index * profile.states.len() + class];
                    let underflow =
                        (raw == 0.0 && !exact_zero) || (raw > 0.0 && (raw as f32) == 0.0);
                    let offset = base + time_index * 7;
                    values[offset] = raw as f32;
                    values[offset + 1] = profile.log_class_size[class] as f32;
                    values[offset + 2] = stationary as f32;
                    values[offset + 3] = (raw.max(self.cache.config.floor).ln()
                        - stationary.max(self.cache.config.floor).ln())
                        as f32;
                    values[offset + 4] = f32::from(exact_zero);
                    values[offset + 5] = f32::from(underflow);
                    values[offset + 6] = f32::from(raw < self.cache.config.floor);
                }
            }
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, n, self.spec.channels],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl DiffusionProducer {
    pub(super) fn new(spec: FeatureSpec, cache: Arc<ProfileCache>) -> Self {
        Self { spec, cache }
    }
}

fn row_weights(state: &MatrixState) -> Vec<usize> {
    let words = state.words_per_row() as usize;
    state
        .packed_rows()
        .chunks_exact(words)
        .map(|row| row.iter().map(|word| word.count_ones() as usize).sum())
        .collect()
}

fn column_weights(state: &MatrixState) -> Vec<usize> {
    let n = state.dimension() as usize;
    (0..n)
        .map(|column| {
            (0..n)
                .map(|row| matrix_bit(state, row, column) as usize)
                .sum()
        })
        .collect()
}

fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> bool {
    let words_per_row = state.words_per_row() as usize;
    let word = state.packed_rows()[row * words_per_row + column / 64];
    ((word >> (column % 64)) & 1) != 0
}

fn pair_histogram(state: &MatrixState, first: usize, second: usize) -> [usize; 4] {
    let n = state.dimension() as usize;
    let mut histogram = [0usize; 4];
    for row in 0..n {
        let pattern = (matrix_bit(state, row, first) as usize)
            | ((matrix_bit(state, row, second) as usize) << 1);
        histogram[pattern] += 1;
    }
    histogram
}
