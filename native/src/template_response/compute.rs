mod output;

use super::assignment::template_assignment_cost;
use super::budget::{source_evaluation_estimate, template_response_workspace_bytes};
use super::provider::CachedTemplateResponse;
use super::TemplateResponseConfig;
use crate::features::FeatureComputation;
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

    output::build_response(
        state,
        &decomposition.source_wires,
        config,
        &assignment_costs,
    )
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
