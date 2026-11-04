use super::super::registry::FeatureRegistry;
use super::super::FeatureValue;
use super::validation::{
    copy_feature_row, element_count, feature_shape, validate_value, FeatureRowTarget,
};
use super::{FeatureBuildError, FeatureComputation, FeaturePolicy};
use crate::packet::{FeatureEntry, FeatureStatus};
use crate::schema::{FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use std::collections::HashMap;

pub(super) struct FeatureEntryContext<'a> {
    pub(super) states: &'a [MatrixState],
    pub(super) nodes: usize,
    pub(super) enabled: bool,
    pub(super) budget_exhausted: bool,
    pub(super) remaining_evaluations: &'a mut Option<usize>,
    pub(super) policy: &'a FeaturePolicy,
    pub(super) structural_rows: &'a [Option<HashMap<FeatureId, FeatureValue>>],
}

impl FeatureRegistry {
    pub(super) fn build_entry(
        &self,
        spec: FeatureSpec,
        context: FeatureEntryContext<'_>,
    ) -> Result<FeatureEntry, FeatureBuildError> {
        let FeatureEntryContext {
            states,
            nodes,
            enabled,
            budget_exhausted,
            remaining_evaluations,
            policy,
            structural_rows,
        } = context;
        let producer = self.providers.get(&spec.id);
        let mut results = Vec::with_capacity(states.len());
        let mut max_groups = 0usize;
        for (state_index, state) in states.iter().enumerate() {
            let result = if !enabled {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::NotRequested,
                    reason: "feature_not_selected_by_policy".into(),
                }
            } else if budget_exhausted {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::ResourceLimit,
                    reason: "workspace_estimate_exceeds_policy_limit".into(),
                }
            } else if let Some(value) = structural_rows
                .get(state_index)
                .and_then(Option::as_ref)
                .and_then(|row| row.get(&spec.id))
            {
                FeatureComputation::Computed(value.clone())
            } else if let Some(producer) = producer {
                if remaining_evaluations.is_some_and(|remaining| remaining == 0) {
                    FeatureComputation::Unavailable {
                        status: FeatureStatus::ResourceLimit,
                        reason: "feature_evaluation_budget_exhausted".into(),
                    }
                } else {
                    if let Some(remaining) = remaining_evaluations {
                        *remaining -= 1;
                    }
                    producer.compute(state, policy)
                }
            } else {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::Unsupported,
                    reason: "no_native_provider_registered".into(),
                }
            };
            if let FeatureComputation::Computed(value)
            | FeatureComputation::ComputedWithProvenance { value, .. } = &result
            {
                validate_value(&spec, state.dimension() as usize, value)?;
                if spec.role == FeatureRole::Group {
                    max_groups = max_groups.max(value.shape[0]);
                }
                if let FeatureComputation::ComputedWithProvenance { provenance, .. } = &result {
                    if provenance.iter().any(|item| item.trim().is_empty()) {
                        return Err(FeatureBuildError(format!(
                            "{} returned empty provenance metadata",
                            spec.id
                        )));
                    }
                }
            } else if let FeatureComputation::Unavailable { status, reason } = &result {
                if *status == FeatureStatus::Computed || reason.trim().is_empty() {
                    return Err(FeatureBuildError(format!(
                        "{} returned invalid unavailable status",
                        spec.id
                    )));
                }
            }
            results.push(result);
        }

        let shape = feature_shape(&spec, states.len(), nodes, max_groups);
        let total_values = element_count(&shape)?;
        let mut values = vec![0.0f32; total_values];
        let mut available = Vec::with_capacity(states.len());
        let mut status = Vec::with_capacity(states.len());
        let mut reason = Vec::with_capacity(states.len());
        let mut provenance = vec![Vec::new(); states.len()];
        let mut token_valid = if spec.role == FeatureRole::Group {
            vec![
                0u8;
                states
                    .len()
                    .checked_mul(max_groups)
                    .ok_or_else(|| FeatureBuildError("group mask size overflow".into()))?
            ]
        } else {
            Vec::new()
        };
        let role_count = spec.wire_roles.len();
        let incidence_count = states
            .len()
            .checked_mul(max_groups)
            .and_then(|value| value.checked_mul(nodes))
            .and_then(|value| value.checked_mul(role_count))
            .ok_or_else(|| FeatureBuildError("group incidence size overflow".into()))?;
        let mut incidence = if spec.role == FeatureRole::Group {
            vec![0.0f32; incidence_count]
        } else {
            Vec::new()
        };

        for (batch_index, (state, result)) in states.iter().zip(results).enumerate() {
            match result {
                FeatureComputation::Unavailable {
                    status: row_status,
                    reason: row_reason,
                } => {
                    available.push(0);
                    status.push(row_status);
                    reason.push(row_reason);
                }
                FeatureComputation::Computed(value) => {
                    available.push(1);
                    status.push(FeatureStatus::Computed);
                    reason.push(String::new());
                    copy_feature_row(
                        &spec,
                        &value,
                        FeatureRowTarget {
                            batch_index,
                            state_nodes: state.dimension() as usize,
                            max_nodes: nodes,
                            max_groups,
                            values: &mut values,
                            token_valid: &mut token_valid,
                            incidence: &mut incidence,
                        },
                    )?;
                }
                FeatureComputation::ComputedWithProvenance {
                    value,
                    provenance: row_provenance,
                } => {
                    available.push(1);
                    status.push(FeatureStatus::Computed);
                    reason.push(String::new());
                    provenance[batch_index] = row_provenance;
                    copy_feature_row(
                        &spec,
                        &value,
                        FeatureRowTarget {
                            batch_index,
                            state_nodes: state.dimension() as usize,
                            max_nodes: nodes,
                            max_groups,
                            values: &mut values,
                            token_valid: &mut token_valid,
                            incidence: &mut incidence,
                        },
                    )?;
                }
            }
        }

        Ok(FeatureEntry {
            spec: spec.clone(),
            shape,
            values,
            available,
            status,
            reason,
            provenance,
            wire_roles: spec.wire_roles.clone(),
            token_valid: (spec.role == FeatureRole::Group).then_some(token_valid),
            incidence: (spec.role == FeatureRole::Group).then_some(incidence),
        })
    }
}
