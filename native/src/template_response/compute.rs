use super::assignment::template_assignment_cost;
use super::budget::{source_evaluation_estimate, template_response_workspace_bytes};
use super::moments::{
    conditional_choice_totals, conditional_round_cost, deterministic_probability,
    expected_cost_fraction, mixed_difference_at_midpoint,
};
use super::provider::{CachedTemplateResponse, TemplateResponseValues};
use super::rational::{rational_to_f32, signed_rational_to_f32};
use super::{RationalProbability, TemplateResponseConfig, TEMPLATE_RESPONSE_WIRE_ROLES};
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use crate::shear::partial_shear;
use crate::state::MatrixState;

pub(super) fn compute_template_response(
    state: &MatrixState,
    config: &TemplateResponseConfig,
) -> CachedTemplateResponse {
    let width = (state.dimension() / 2) as usize;
    let choice_count = config.choices.len();
    let Some(assignment_count) = 1usize.checked_shl(choice_count as u32) else {
        return cached_resource_limit("template_response_assignment_count_overflow");
    };
    if width > config.budgets.max_width {
        return cached_resource_limit("template_response_width_budget_exhausted");
    }
    if assignment_count > config.budgets.max_assignments {
        return cached_resource_limit("template_response_assignment_budget_exhausted");
    }
    let Some(workspace_bytes) = template_response_workspace_bytes(state.dimension(), choice_count)
    else {
        return cached_resource_limit("template_response_workspace_estimate_overflow");
    };
    if workspace_bytes > config.budgets.max_bytes {
        return cached_resource_limit("template_response_workspace_budget_exhausted");
    }
    let Some(required_evaluations) =
        source_evaluation_estimate(choice_count, config.points.len(), assignment_count)
    else {
        return cached_resource_limit("template_response_evaluation_estimate_overflow");
    };
    if required_evaluations > config.budgets.max_evaluations {
        return cached_resource_limit("template_response_evaluation_budget_exhausted");
    }

    let decomposition = match partial_shear(state) {
        Ok(Some(value)) => value,
        Ok(None) => {
            return cached_inapplicable("no_nonzero_partial_shear_block");
        }
        Err(_) => return cached_unsupported("partial_shear_decomposition_failed"),
    };
    if config.choices.iter().any(|choice| {
        choice.control >= decomposition.source_wires.len()
            || choice.target >= decomposition.source_wires.len()
    }) {
        return cached_unsupported("template_choice_outside_partial_shear_source_block");
    }

    let mut assignment_costs = Vec::with_capacity(assignment_count);
    for assignment in 0..assignment_count {
        let Some(cost) =
            template_assignment_cost(&decomposition.block, &config.choices, assignment)
        else {
            return cached_resource_limit("template_response_cost_overflow");
        };
        assignment_costs.push(cost);
    }

    let midpoint = vec![
        RationalProbability {
            numerator: 1,
            denominator: 2
        };
        choice_count
    ];
    let Some((e02_numerator, e02_denominator)) =
        expected_cost_fraction(&assignment_costs, &midpoint)
    else {
        return cached_resource_limit("template_response_exact_rational_overflow");
    };
    let Some(e02_value) = rational_to_f32(e02_numerator, e02_denominator) else {
        return cached_resource_limit("template_response_float32_conversion_failed");
    };

    let mut e03_values = Vec::with_capacity(config.points.len());
    for point in &config.points {
        let Some((numerator, denominator)) = expected_cost_fraction(&assignment_costs, point)
        else {
            return cached_resource_limit("template_response_exact_rational_overflow");
        };
        let Some(value) = rational_to_f32(numerator, denominator) else {
            return cached_resource_limit("template_response_float32_conversion_failed");
        };
        e03_values.push(value);
    }

    let mut e06_values = Vec::new();
    let mut e06_incidence = Vec::new();
    for first in 0..choice_count {
        for second in first + 1..choice_count {
            let Some((numerator, denominator)) =
                mixed_difference_at_midpoint(&assignment_costs, first, second)
            else {
                return cached_resource_limit("template_response_mixed_difference_overflow");
            };
            let Some(value) = signed_rational_to_f32(numerator, denominator) else {
                return cached_resource_limit("template_response_float32_conversion_failed");
            };
            e06_values.push(value);

            let wires = [
                decomposition.source_wires[config.choices[first].control],
                decomposition.source_wires[config.choices[first].target],
                decomposition.source_wires[config.choices[second].control],
                decomposition.source_wires[config.choices[second].target],
            ];
            for node in 0..state.dimension() as usize {
                for role in 0..TEMPLATE_RESPONSE_WIRE_ROLES.len() {
                    e06_incidence.push(if wires[role] == node { 1.0 } else { 0.0 });
                }
            }
        }
    }

    let Some(e07_cost) = conditional_round_cost(&assignment_costs, &midpoint) else {
        return cached_resource_limit("template_response_exact_rational_overflow");
    };
    let mut reverse_point = midpoint.clone();
    for index in (0..choice_count).rev() {
        let Some((zero_cost, one_cost)) =
            conditional_choice_totals(&assignment_costs, &reverse_point, index)
        else {
            return cached_resource_limit("template_response_exact_rational_overflow");
        };
        reverse_point[index] = deterministic_probability(one_cost < zero_cost);
    }
    let reverse_bits = reverse_point
        .iter()
        .enumerate()
        .fold(0usize, |mask, (index, probability)| {
            mask | (usize::from(probability.numerator == 1) << index)
        });
    let reverse_cost = assignment_costs[reverse_bits];

    let best_cost = *assignment_costs.iter().min().expect("nonempty assignments");
    let mut multistart_costs = Vec::with_capacity(config.points.len() + 1);
    multistart_costs.push(e07_cost);
    for point in &config.points {
        let Some(cost) = conditional_round_cost(&assignment_costs, point) else {
            return cached_resource_limit("template_response_exact_rational_overflow");
        };
        multistart_costs.push(cost);
    }
    let min_multistart = *multistart_costs
        .iter()
        .min()
        .expect("midpoint start exists");
    let max_multistart = *multistart_costs
        .iter()
        .max()
        .expect("midpoint start exists");

    CachedTemplateResponse::Computed(TemplateResponseValues {
        e02: scalar_feature(e02_value),
        e03: FeatureValue {
            shape: vec![e03_values.len()],
            values: e03_values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
        e06: FeatureValue {
            shape: vec![e06_values.len(), 1],
            values: e06_values,
            token_valid: vec![1; choice_count * (choice_count - 1) / 2],
            incidence: e06_incidence,
        },
        e07: scalar_feature(e07_cost as f32),
        e08: FeatureValue {
            shape: vec![2],
            values: vec![best_cost as f32, 1.0],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
        e09: FeatureValue {
            shape: vec![5],
            values: vec![
                e07_cost as f32,
                reverse_cost as f32,
                e07_cost.abs_diff(reverse_cost) as f32,
                min_multistart as f32,
                max_multistart as f32,
            ],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
    })
}

fn scalar_feature(value: f32) -> FeatureValue {
    FeatureValue {
        shape: vec![1],
        values: vec![value],
        token_valid: Vec::new(),
        incidence: Vec::new(),
    }
}

fn cached_unsupported(reason: &str) -> CachedTemplateResponse {
    CachedTemplateResponse::Unavailable(FeatureStatus::Unsupported, reason.into())
}

fn cached_inapplicable(reason: &str) -> CachedTemplateResponse {
    CachedTemplateResponse::Unavailable(FeatureStatus::Inapplicable, reason.into())
}

fn cached_resource_limit(reason: &str) -> CachedTemplateResponse {
    CachedTemplateResponse::Unavailable(FeatureStatus::ResourceLimit, reason.into())
}

pub(super) fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

pub(super) fn resource_limit(reason: &str) -> FeatureComputation {
    unavailable(FeatureStatus::ResourceLimit, reason)
}
