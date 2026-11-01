//! Fail-closed feature registry and padded packet assembly.
//!
//! Provider code is deliberately absent until an exact Rust implementation is
//! ported and reviewed. Missing providers are recorded as unavailable; they are
//! never replaced by zero-valued features marked as computed.

use crate::packet::{FeatureEntry, FeaturePacket, FeaturePacketError, FeatureStatus};
use crate::schema::{
    all_feature_specs, feature_spec, FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT,
};
use crate::state::MatrixState;
use crate::structural;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureBuildError(pub String);

impl fmt::Display for FeatureBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FeatureBuildError {}

impl From<FeaturePacketError> for FeatureBuildError {
    fn from(error: FeaturePacketError) -> Self {
        Self(error.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePolicy {
    pub enabled_ids: Vec<FeatureId>,
    pub seed: u64,
    pub schema_id: String,
    pub solver_assisted: bool,
    pub symmetry_contract: String,
    pub evaluations: Option<usize>,
    pub max_bytes: Option<usize>,
}

impl FeaturePolicy {
    pub fn complete_103(schema_id: impl Into<String>) -> Self {
        Self {
            enabled_ids: all_feature_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect(),
            seed: 0,
            schema_id: schema_id.into(),
            solver_assisted: false,
            symmetry_contract: "exact_relabel".into(),
            evaluations: None,
            max_bytes: None,
        }
    }

    pub fn id(&self) -> Result<String, FeatureBuildError> {
        validate_policy(self)?;
        // Stable, length-delimited textual form suitable for logs and checkpoint binding.
        // The campaign layer adds a cryptographic digest when it commits artifacts.
        let mut body = String::new();
        push_field(&mut body, &self.schema_id);
        push_field(&mut body, &self.seed.to_string());
        push_field(
            &mut body,
            if self.solver_assisted {
                "solver=1"
            } else {
                "solver=0"
            },
        );
        push_field(&mut body, &self.symmetry_contract);
        push_field(
            &mut body,
            &self
                .evaluations
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        push_field(
            &mut body,
            &self
                .max_bytes
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        for id in &self.enabled_ids {
            push_field(&mut body, &id.to_string());
        }
        Ok(body)
    }
}

fn push_field(output: &mut String, value: &str) {
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}

fn validate_policy(policy: &FeaturePolicy) -> Result<(), FeatureBuildError> {
    if policy.enabled_ids.is_empty() {
        return Err(FeatureBuildError(
            "feature policy selects no feature IDs".into(),
        ));
    }
    if policy.schema_id.trim().is_empty() || policy.symmetry_contract.trim().is_empty() {
        return Err(FeatureBuildError(
            "feature policy needs schema and symmetry identities".into(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(policy.enabled_ids.len());
    if policy.enabled_ids.iter().any(|id| !seen.insert(*id)) {
        return Err(FeatureBuildError(
            "feature policy repeats a feature ID".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct FeatureValue {
    /// Shape excluding the batch axis.
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    /// Group-only metadata. `incidence` is `[groups, nodes, wire_roles]`.
    pub token_valid: Vec<u8>,
    pub incidence: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum FeatureComputation {
    Computed(FeatureValue),
    Unavailable {
        status: FeatureStatus,
        reason: String,
    },
}

pub trait FeatureProducer: Send + Sync {
    fn spec(&self) -> &FeatureSpec;

    /// Names the actual symmetry contract implemented by the provider.
    fn symmetry_contract(&self) -> &str;

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation;
}

pub struct FeatureRegistry {
    specs: HashMap<FeatureId, FeatureSpec>,
    providers: HashMap<FeatureId, Box<dyn FeatureProducer>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut specs = all_feature_specs();
        for spec in &mut specs {
            if let Some(metadata) = structural::metadata(spec.id) {
                spec.scope = metadata.scope.into();
                spec.producer_version = metadata.producer_version.into();
                spec.dependencies = metadata
                    .dependencies
                    .iter()
                    .map(|value| (*value).into())
                    .collect();
                spec.exactness = metadata.exactness.into();
                spec.cost_tier = metadata.cost_tier.into();
                spec.wire_roles.clear();
            }
        }
        Self {
            specs: specs.into_iter().map(|spec| (spec.id, spec)).collect(),
            providers: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        producer: Box<dyn FeatureProducer>,
    ) -> Result<(), FeatureBuildError> {
        let spec = producer.spec().clone();
        let canonical = feature_spec(spec.id);
        if spec.role != canonical.role || spec.channels != canonical.channels {
            return Err(FeatureBuildError(format!(
                "{} provider differs from canonical layout",
                spec.id
            )));
        }
        if spec.producer_version.trim().is_empty() || spec.producer_version == "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} provider has no version",
                spec.id
            )));
        }
        if spec.requires_solver_assisted && producer.symmetry_contract().trim().is_empty() {
            return Err(FeatureBuildError(format!(
                "{} solver-assisted provider lacks a contract",
                spec.id
            )));
        }
        if self.providers.contains_key(&spec.id) {
            return Err(FeatureBuildError(format!(
                "{} provider already registered",
                spec.id
            )));
        }
        if self
            .specs
            .get(&spec.id)
            .is_some_and(|existing| existing.producer_version != "unconfigured")
        {
            return Err(FeatureBuildError(format!(
                "{} already has a built-in provider",
                spec.id
            )));
        }
        self.specs.insert(spec.id, spec.clone());
        self.providers.insert(spec.id, producer);
        Ok(())
    }

    /// Returns specs in the canonical 103-ID order, including the actual
    /// producer metadata used by this registry.
    pub fn specs(&self) -> Vec<FeatureSpec> {
        (0..FEATURE_COUNT)
            .filter_map(FeatureId::from_index)
            .map(|id| {
                self.specs
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| feature_spec(id))
            })
            .collect()
    }

    pub fn build(
        &self,
        states: &[MatrixState],
        policy: &FeaturePolicy,
    ) -> Result<FeaturePacket, FeatureBuildError> {
        validate_policy(policy)?;
        if states.is_empty() {
            return Err(FeatureBuildError(
                "cannot build an empty feature batch".into(),
            ));
        }
        for (index, state) in states.iter().enumerate() {
            if state.alphabet_id().trim().is_empty() {
                return Err(FeatureBuildError(format!(
                    "state {index} has no alphabet identity"
                )));
            }
        }
        for id in &policy.enabled_ids {
            let spec = self
                .specs
                .get(id)
                .ok_or_else(|| FeatureBuildError(format!("unknown feature ID {id}")))?;
            if spec.requires_solver_assisted && !policy.solver_assisted {
                return Err(FeatureBuildError(format!(
                    "{id} requires a solver-assisted feature policy"
                )));
            }
            if policy.symmetry_contract == "exact_relabel" {
                if let Some(provider) = self.providers.get(id) {
                    if provider.symmetry_contract() != "exact_relabel" {
                        return Err(FeatureBuildError(format!(
                            "{id} provider symmetry contract mismatch"
                        )));
                    }
                }
            }
        }

        let nodes = states
            .iter()
            .map(|state| state.dimension() as usize)
            .max()
            .unwrap_or(0);
        let valid_nodes = states
            .iter()
            .flat_map(|state| {
                (0..nodes).map(move |node| u8::from(node < state.dimension() as usize))
            })
            .collect::<Vec<_>>();
        let estimated_bytes = states
            .iter()
            .try_fold(0usize, |total, state| {
                let n = state.dimension() as usize;
                n.checked_mul(n)
                    .and_then(|square| square.checked_mul(1024))
                    .and_then(|matrix| {
                        n.checked_mul(256)
                            .and_then(|vectors| matrix.checked_add(vectors))
                    })
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| FeatureBuildError("feature workspace estimate overflow".into()))?;
        let budget_exhausted_before_compute = policy
            .max_bytes
            .is_some_and(|limit| estimated_bytes > limit);

        // The source implementation computes its structural bundle once per
        // state; selected IDs are read from the shared result.
        let needs_structural = !budget_exhausted_before_compute
            && policy
                .enabled_ids
                .iter()
                .any(|id| structural::is_builtin(*id));
        let structural_rows = if needs_structural {
            states
                .iter()
                .map(|state| {
                    structural::compute(state, &policy.enabled_ids)
                        .map(Some)
                        .map_err(FeatureBuildError)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; states.len()]
        };

        let enabled = policy
            .enabled_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut remaining_evaluations = policy.evaluations;
        let mut entries = Vec::with_capacity(FEATURE_COUNT);
        for index in 0..FEATURE_COUNT {
            let id = FeatureId::from_index(index)
                .ok_or_else(|| FeatureBuildError("feature manifest index invalid".into()))?;
            let spec = self
                .specs
                .get(&id)
                .cloned()
                .unwrap_or_else(|| feature_spec(id));
            entries.push(self.build_entry(
                spec,
                states,
                nodes,
                enabled.contains(&id),
                budget_exhausted_before_compute,
                &mut remaining_evaluations,
                policy,
                &structural_rows,
            )?);
        }
        let policy_id = policy.id()?;
        FeaturePacket::new(
            entries,
            states,
            nodes,
            valid_nodes,
            policy.schema_id.clone(),
            policy_id,
        )
        .map_err(FeatureBuildError::from)
    }

    fn build_entry(
        &self,
        spec: FeatureSpec,
        states: &[MatrixState],
        nodes: usize,
        enabled: bool,
        budget_exhausted: bool,
        remaining_evaluations: &mut Option<usize>,
        policy: &FeaturePolicy,
        structural_rows: &[Option<HashMap<FeatureId, FeatureValue>>],
    ) -> Result<FeatureEntry, FeatureBuildError> {
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
            } else if producer.is_none() {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::Unsupported,
                    reason: "no_native_provider_registered".into(),
                }
            } else if remaining_evaluations.is_some_and(|remaining| remaining == 0) {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::ResourceLimit,
                    reason: "feature_evaluation_budget_exhausted".into(),
                }
            } else {
                if let Some(remaining) = remaining_evaluations {
                    *remaining -= 1;
                }
                producer.expect("checked above").compute(state, policy)
            };
            if let FeatureComputation::Computed(value) = &result {
                validate_value(&spec, state.dimension() as usize, value)?;
                if spec.role == FeatureRole::Group {
                    max_groups = max_groups.max(value.shape[0]);
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
                        batch_index,
                        state.dimension() as usize,
                        nodes,
                        max_groups,
                        &mut values,
                        &mut token_valid,
                        &mut incidence,
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
            wire_roles: spec.wire_roles.clone(),
            token_valid: (spec.role == FeatureRole::Group).then_some(token_valid),
            incidence: (spec.role == FeatureRole::Group).then_some(incidence),
        })
    }
}

fn validate_value(
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

fn feature_shape(spec: &FeatureSpec, batch: usize, nodes: usize, groups: usize) -> Vec<usize> {
    match spec.role {
        FeatureRole::Node => vec![batch, nodes, spec.channels],
        FeatureRole::Pair => vec![batch, nodes, nodes, spec.channels],
        FeatureRole::Global => vec![batch, spec.channels],
        FeatureRole::Group => vec![batch, groups, spec.channels],
    }
}

fn element_count(shape: &[usize]) -> Result<usize, FeatureBuildError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| FeatureBuildError("feature tensor element count overflow".into()))
}

fn copy_feature_row(
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
