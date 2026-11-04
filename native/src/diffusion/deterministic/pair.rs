//! Ordered-pair diffusion projections.

use super::super::profiles::pair_state_count;
use super::super::unavailable;
use super::weights::pair_histogram;
use super::DiffusionProducer;
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use crate::state::MatrixState;

pub(super) fn compute(
    producer: &DiffusionProducer,
    state: &MatrixState,
    n: usize,
) -> FeatureComputation {
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
    if state_count > producer.cache.config.max_pair_states {
        return unavailable(
            FeatureStatus::ResourceLimit,
            format!(
                "pair diffusion resource_limit: {state_count} states exceeds {}",
                producer.cache.config.max_pair_states
            ),
        );
    }
    let Some(profile) = producer.cache.get_pair(n) else {
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
    let Some(value_count) = pair_count.checked_mul(producer.spec.channels) else {
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
            let base = (first * n + second) * producer.spec.channels;
            for time_index in 0..4 {
                let raw = profile.raw(time_index, class);
                let stationary = profile.stationary[class];
                let exact_zero = profile.exact_zero[time_index * profile.states.len() + class];
                let underflow = (raw == 0.0 && !exact_zero) || (raw > 0.0 && (raw as f32) == 0.0);
                let offset = base + time_index * 7;
                values[offset] = raw as f32;
                values[offset + 1] = profile.log_class_size[class] as f32;
                values[offset + 2] = stationary as f32;
                values[offset + 3] = (raw.max(producer.cache.config.floor).ln()
                    - stationary.max(producer.cache.config.floor).ln())
                    as f32;
                values[offset + 4] = f32::from(exact_zero);
                values[offset + 5] = f32::from(underflow);
                values[offset + 6] = f32::from(raw < producer.cache.config.floor);
            }
        }
    }
    FeatureComputation::Computed(FeatureValue {
        shape: vec![n, n, producer.spec.channels],
        values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}
