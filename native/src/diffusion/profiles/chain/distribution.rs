//! Uniformized transition propagation and Poisson truncation.

use super::super::super::{MAX_UNIFORMIZATION_TERMS, POISSON_TAIL_MARGIN, POISSON_TAIL_SIGMAS};

pub(super) fn distribution(
    transitions: &[Vec<(usize, f64)>],
    initial: usize,
    time: f64,
) -> Option<Vec<f64>> {
    if time == 0.0 {
        let mut mass = vec![0.0; transitions.len()];
        *mass.get_mut(initial)? = 1.0;
        return Some(mass);
    }
    let weights = normalized_poisson_weights(time)?;
    let mut current = vec![0.0; transitions.len()];
    *current.get_mut(initial)? = 1.0;
    let mut mass = vec![0.0; transitions.len()];
    for (step, weight) in weights.iter().copied().enumerate() {
        if weight != 0.0 {
            for (total, value) in mass.iter_mut().zip(&current) {
                *total += weight * value;
            }
        }
        if step + 1 < weights.len() {
            let mut next = vec![0.0; transitions.len()];
            for (source, value) in current.iter().copied().enumerate() {
                if value == 0.0 {
                    continue;
                }
                for &(destination, probability) in &transitions[source] {
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

pub(super) fn normalized_poisson_weights(time: f64) -> Option<Vec<f64>> {
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
