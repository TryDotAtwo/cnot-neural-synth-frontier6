//! Exact-minimum comparison for the explicitly approximate K=4 Bellman subset.
//! This module is pure bookkeeping; real per-state values must come from one
//! pinned checkpoint and one sealed root before these metrics are meaningful.

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub(crate) struct SampledVsFullBellmanAudit {
    pub total_legal_actions: usize,
    pub sampled_actions: usize,
    pub full_minimum: f64,
    pub sampled_minimum: f64,
    pub gap_cost_plus_v: f64,
    pub full_minimizer_count: usize,
    pub sampled_minimizer_count: usize,
    pub sampled_contains_minimizer: bool,
}

/// Compare sampled g+V(child) values with the exact full legal-action backup.
///
/// The caller supplies each sampled action's alphabet ordinal and its score
/// under the same model weights/root as the full vector. tolerance is an
/// absolute tie and same-score tolerance in cost+V units.
pub(crate) fn compare_sampled_vs_full_bellman(
    full_totals: &[f64],
    sampled_indices: &[i64],
    sampled_totals: &[f64],
    tolerance: f64,
) -> Result<SampledVsFullBellmanAudit, String> {
    if full_totals.is_empty()
        || sampled_indices.is_empty()
        || sampled_indices.len() != sampled_totals.len()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || tolerance > 1.0
        || full_totals.iter().chain(sampled_totals).any(|value| !value.is_finite())
    {
        return Err("Bellman audit inputs are empty, nonfinite, or mismatched".into());
    }
    if sampled_indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("sampled Bellman action ordinals must be distinct and alphabet-sorted".into());
    }
    for (&index, &sampled) in sampled_indices.iter().zip(sampled_totals) {
        let exact = usize::try_from(index)
            .ok()
            .and_then(|position| full_totals.get(position))
            .ok_or("sampled Bellman action ordinal is outside the legal alphabet")?;
        if (sampled - exact).abs() > tolerance {
            return Err("sampled and full Bellman scores differ for the same action".into());
        }
    }

    let full_minimum = full_totals.iter().copied().fold(f64::INFINITY, f64::min);
    let sampled_minimum = sampled_totals.iter().copied().fold(f64::INFINITY, f64::min);
    if sampled_minimum + tolerance < full_minimum {
        return Err("sampled Bellman minimum is below the full legal-action minimum".into());
    }
    let full_minimizer_count = full_totals
        .iter()
        .filter(|value| (**value - full_minimum).abs() <= tolerance)
        .count();
    let sampled_minimizer_count = sampled_indices
        .iter()
        .zip(sampled_totals)
        .filter(|(_, value)| (**value - full_minimum).abs() <= tolerance)
        .count();

    Ok(SampledVsFullBellmanAudit {
        total_legal_actions: full_totals.len(),
        sampled_actions: sampled_indices.len(),
        full_minimum,
        sampled_minimum,
        gap_cost_plus_v: (sampled_minimum - full_minimum).max(0.0),
        full_minimizer_count,
        sampled_minimizer_count,
        sampled_contains_minimizer: sampled_minimizer_count > 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bellman::sampling::sampled_indices;
    use tch::{Device, Tensor};

    #[test]
    fn k4_top2_plus_seeded_exploration_reports_gap_to_full_minimum() {
        let qhat = Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]);
        let indices = Vec::<i64>::try_from(sampled_indices(&qhat, 4, 0).unwrap().to_device(Device::Cpu))
            .unwrap();
        assert_eq!(indices, vec![0, 1, 5, 10]);

        let full = [5.0, 6.0, 4.0, 0.5, 3.0, 2.0, 7.0, 8.0, 4.5, 6.0, 1.5, 2.5];
        let sampled: Vec<_> = indices.iter().map(|&index| full[index as usize]).collect();
        let audit = compare_sampled_vs_full_bellman(&full, &indices, &sampled, 1e-9).unwrap();

        assert_eq!(audit.total_legal_actions, 12);
        assert_eq!(audit.sampled_actions, 4);
        assert_eq!(audit.full_minimum, 0.5);
        assert_eq!(audit.sampled_minimum, 1.5);
        assert_eq!(audit.gap_cost_plus_v, 1.0);
        assert_eq!(audit.full_minimizer_count, 1);
        assert_eq!(audit.sampled_minimizer_count, 0);
        assert!(!audit.sampled_contains_minimizer);
    }

    #[test]
    fn tied_full_minima_are_recalled_if_any_tied_action_is_sampled() {
        let mut full = vec![4.0; 12];
        full[10] = 1.0;
        full[11] = 1.0;
        let indices = vec![0, 1, 5, 10];
        let sampled: Vec<_> = indices.iter().map(|&index| full[index as usize]).collect();
        let audit = compare_sampled_vs_full_bellman(&full, &indices, &sampled, 0.0).unwrap();
        assert_eq!(audit.full_minimizer_count, 2);
        assert_eq!(audit.sampled_minimizer_count, 1);
        assert!(audit.sampled_contains_minimizer);
        assert_eq!(audit.gap_cost_plus_v, 0.0);
    }

    #[test]
    fn malformed_or_nonmatching_sampled_scores_are_rejected() {
        let full = [1.0, 2.0, 3.0];
        assert!(compare_sampled_vs_full_bellman(&full, &[0, 0], &[1.0, 1.0], 0.0).is_err());
        assert!(compare_sampled_vs_full_bellman(&full, &[3], &[1.0], 0.0).is_err());
        assert!(compare_sampled_vs_full_bellman(&full, &[1], &[2.5], 0.0).is_err());
        assert!(compare_sampled_vs_full_bellman(&full, &[1], &[f64::NAN], 0.0).is_err());
    }
}
