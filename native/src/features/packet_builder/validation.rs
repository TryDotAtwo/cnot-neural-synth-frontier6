use super::super::{FeatureBuildError, FeatureValue};
use crate::schema::{FeatureRole, FeatureSpec};

pub(super) fn validate_value(
    spec: &FeatureSpec,
    nodes: usize,
    value: &FeatureValue,
) -> Result<(), FeatureBuildError> {
    let expected_rank = match spec.role {
        FeatureRole::Node | FeatureRole::Group => 2,
        FeatureRole::Pair => 3,
        FeatureRole::Global => 1,
    };
    if value.shape.len() != expected_rank || value.shape.last().copied() != Some(spec.channels) {
        return Err(FeatureBuildError(format!(
            "{} provider returned the wrong rank or channel count",
            spec.id
        )));
    }
    let expected_values = element_count(&value.shape)?;
    if value.values.len() != expected_values || value.values.iter().any(|item| !item.is_finite()) {
        return Err(FeatureBuildError(format!(
            "{} provider returned invalid/non-finite values",
            spec.id
        )));
    }
    match spec.role {
        FeatureRole::Node if value.shape[0] != nodes => {
            return Err(FeatureBuildError(format!(
                "{} node extent differs from state",
                spec.id
            )));
        }
        FeatureRole::Pair if value.shape[0] != nodes || value.shape[1] != nodes => {
            return Err(FeatureBuildError(format!(
                "{} pair extent differs from state",
                spec.id
            )));
        }
        FeatureRole::Global => {
            if !value.token_valid.is_empty() || !value.incidence.is_empty() {
                return Err(FeatureBuildError(format!(
                    "{} global feature returned group metadata",
                    spec.id
                )));
            }
        }
        FeatureRole::Group => {
            let groups = value.shape[0];
            let expected_incidence = groups
                .checked_mul(nodes)
                .and_then(|extent| extent.checked_mul(spec.wire_roles.len()))
                .ok_or_else(|| FeatureBuildError("group incidence length overflow".into()))?;
            if value.token_valid.len() != groups
                || value.token_valid.iter().any(|mask| *mask > 1)
                || value.incidence.len() != expected_incidence
                || value
                    .incidence
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(FeatureBuildError(format!(
                    "{} group metadata shape/value mismatch",
                    spec.id
                )));
            }
        }
        _ => {
            if !value.token_valid.is_empty() || !value.incidence.is_empty() {
                return Err(FeatureBuildError(format!(
                    "{} non-group feature returned group metadata",
                    spec.id
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn feature_shape(
    spec: &FeatureSpec,
    batch: usize,
    nodes: usize,
    groups: usize,
) -> Vec<usize> {
    match spec.role {
        FeatureRole::Node => vec![batch, nodes, spec.channels],
        FeatureRole::Pair => vec![batch, nodes, nodes, spec.channels],
        FeatureRole::Global => vec![batch, spec.channels],
        FeatureRole::Group => vec![batch, groups, spec.channels],
    }
}

pub(super) fn element_count(shape: &[usize]) -> Result<usize, FeatureBuildError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| FeatureBuildError("feature tensor element count overflow".into()))
}

pub(super) fn copy_feature_row(
    spec: &FeatureSpec,
    value: &FeatureValue,
    batch_index: usize,
    state_nodes: usize,
    max_nodes: usize,
    max_groups: usize,
    values: &mut [f32],
    token_valid: &mut [u8],
    incidence: &mut [f32],
) -> Result<(), FeatureBuildError> {
    match spec.role {
        FeatureRole::Node | FeatureRole::Global => {
            let per_row = if spec.role == FeatureRole::Node {
                max_nodes * spec.channels
            } else {
                spec.channels
            };
            let base = batch_index * per_row;
            values[base..base + value.values.len()].copy_from_slice(&value.values);
        }
        FeatureRole::Pair => {
            let source_dim = state_nodes * state_nodes * spec.channels;
            let source_index = |row: usize, col: usize, channel: usize| {
                (row * state_nodes + col) * spec.channels + channel
            };
            let target_index = |row: usize, col: usize, channel: usize| {
                ((batch_index * max_nodes + row) * max_nodes + col) * spec.channels + channel
            };
            if value.values.len() != source_dim {
                return Err(FeatureBuildError(format!(
                    "{} pair source buffer changed after validation",
                    spec.id
                )));
            }
            for row in 0..state_nodes {
                for col in 0..state_nodes {
                    for channel in 0..spec.channels {
                        values[target_index(row, col, channel)] =
                            value.values[source_index(row, col, channel)];
                    }
                }
            }
        }
        FeatureRole::Group => {
            let groups = value.shape[0];
            let channels = spec.channels;
            let target_base = batch_index * max_groups * channels;
            values[target_base..target_base + value.values.len()].copy_from_slice(&value.values);
            let token_base = batch_index * max_groups;
            token_valid[token_base..token_base + groups].copy_from_slice(&value.token_valid);
            let roles = spec.wire_roles.len();
            for group in 0..groups {
                for node in 0..state_nodes {
                    for role in 0..roles {
                        let source = (group * state_nodes + node) * roles + role;
                        let target =
                            ((batch_index * max_groups + group) * max_nodes + node) * roles + role;
                        incidence[target] = value.incidence[source];
                    }
                }
            }
        }
    }
    Ok(())
}
