//! Column-class diffusion projections.

use super::super::unavailable;
use super::weights::{column_weights, matrix_bit, row_weights};
use super::DiffusionProducer;
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use crate::state::MatrixState;

pub(super) fn compute(
    producer: &DiffusionProducer,
    state: &MatrixState,
    n: usize,
) -> FeatureComputation {
    if n > producer.cache.config.max_n as usize {
        return unavailable(
            FeatureStatus::ResourceLimit,
            format!(
                "column diffusion max_n={} exceeded by n={n}",
                producer.cache.config.max_n
            ),
        );
    }
    let Some(profile) = producer.cache.get(n) else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "column diffusion checked state/iteration budget exceeded".into(),
        );
    };
    let Some(value_count) = n.checked_mul(producer.spec.channels) else {
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
        let weight = if producer.spec.id.ordinal() == 2 {
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
        let base = wire * producer.spec.channels;
        match producer.spec.id.ordinal() {
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
                    values[offset + 6] = f32::from(sample.raw < producer.cache.config.floor);
                }
            }
            3 => {
                for time_index in 0..4 {
                    let sample = profile.sample(time_index, class);
                    let offset = base + time_index * 4;
                    values[offset] = sample.log_mass as f32;
                    values[offset + 1] = f32::from(sample.exact_zero);
                    values[offset + 2] = f32::from(sample.underflow);
                    values[offset + 3] = f32::from(sample.raw < producer.cache.config.floor);
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
        shape: vec![n, producer.spec.channels],
        values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}
