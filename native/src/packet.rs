use crate::schema::{FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID};
use crate::state::MatrixState;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureStatus {
    Computed,
    Inapplicable,
    Unsupported,
    ResourceLimit,
    NotRequested,
}

impl FeatureStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Computed => "computed",
            Self::Inapplicable => "inapplicable",
            Self::Unsupported => "unsupported",
            Self::ResourceLimit => "resource_limit",
            Self::NotRequested => "not_requested",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePacketError(pub String);

impl fmt::Display for FeaturePacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FeaturePacketError {}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateIdentity {
    pub dimension: u32,
    pub alphabet_id: String,
    pub packed_rows: Vec<u64>,
}

impl From<&MatrixState> for StateIdentity {
    fn from(state: &MatrixState) -> Self {
        Self {
            dimension: state.dimension(),
            alphabet_id: state.alphabet_id().to_owned(),
            packed_rows: state.packed_rows().to_vec(),
        }
    }
}

/// Dense values are storage only: an unavailable row is invisible to consumers
/// unless its availability byte is set. The zero-filled row is never evidence.
#[derive(Clone, Debug)]
pub struct FeatureEntry {
    pub spec: FeatureSpec,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub available: Vec<u8>,
    pub status: Vec<FeatureStatus>,
    pub reason: Vec<String>,
    /// Per-row audit values; kept outside the numeric tensor used by models.
    pub provenance: Vec<Vec<String>>,
    pub wire_roles: Vec<String>,
    pub token_valid: Option<Vec<u8>>,
    pub incidence: Option<Vec<f32>>,
}

impl FeatureEntry {
    pub fn validate(&self, batch: usize, nmax: usize) -> Result<(), FeaturePacketError> {
        if self.spec.channels == 0 {
            return Err(FeaturePacketError(format!(
                "{} must declare at least one channel",
                self.spec.id
            )));
        }
        if self.spec.scope.trim().is_empty()
            || self.spec.producer_version.trim().is_empty()
            || self.spec.exactness.trim().is_empty()
            || self.spec.cost_tier.trim().is_empty()
        {
            return Err(FeaturePacketError(format!(
                "{} is missing producer identity metadata",
                self.spec.id
            )));
        }
        if self
            .spec
            .dependencies
            .iter()
            .any(|dependency| dependency.trim().is_empty())
        {
            return Err(FeaturePacketError(format!(
                "{} has an empty dependency identity",
                self.spec.id
            )));
        }
        let rank = match self.spec.role {
            FeatureRole::Node | FeatureRole::Group => 3,
            FeatureRole::Pair => 4,
            FeatureRole::Global => 2,
        };
        if self.shape.len() != rank || self.shape.first().copied() != Some(batch) {
            return Err(FeaturePacketError(format!(
                "{} has an invalid feature shape",
                self.spec.id
            )));
        }
        if self.shape.last().copied() != Some(self.spec.channels) {
            return Err(FeaturePacketError(format!(
                "{} channel count differs from schema",
                self.spec.id
            )));
        }
        match self.spec.role {
            FeatureRole::Node if self.shape[1] != nmax => {
                return Err(FeaturePacketError(format!(
                    "{} node width differs from batch",
                    self.spec.id
                )));
            }
            FeatureRole::Pair if self.shape[1] != nmax || self.shape[2] != nmax => {
                return Err(FeaturePacketError(format!(
                    "{} pair width differs from batch",
                    self.spec.id
                )));
            }
            FeatureRole::Group => {
                if self.wire_roles != self.spec.wire_roles
                    || self
                        .spec
                        .wire_roles
                        .iter()
                        .any(|role| role.trim().is_empty())
                    || self
                        .spec
                        .wire_roles
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != self.spec.wire_roles.len()
                {
                    return Err(FeaturePacketError(format!(
                        "{} group role names differ from its schema",
                        self.spec.id
                    )));
                }
                let groups = self.shape[1];
                let incidence_len = batch
                    .checked_mul(groups)
                    .and_then(|x| x.checked_mul(nmax))
                    .and_then(|x| x.checked_mul(self.spec.wire_roles.len()))
                    .ok_or_else(|| FeaturePacketError("group incidence size overflow".into()))?;
                let token_len = batch
                    .checked_mul(groups)
                    .ok_or_else(|| FeaturePacketError("group token mask size overflow".into()))?;
                if self.token_valid.as_ref().map(Vec::len) != Some(token_len)
                    || self.incidence.as_ref().map(Vec::len) != Some(incidence_len)
                {
                    return Err(FeaturePacketError(format!(
                        "{} group metadata shape mismatch",
                        self.spec.id
                    )));
                }
            }
            _ => {
                if !self.spec.wire_roles.is_empty()
                    || !self.wire_roles.is_empty()
                    || self.token_valid.is_some()
                    || self.incidence.is_some()
                {
                    return Err(FeaturePacketError(format!(
                        "{} carries group-only metadata",
                        self.spec.id
                    )));
                }
            }
        }
        let expected_values = self
            .shape
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .ok_or_else(|| FeaturePacketError(format!("{} value size overflow", self.spec.id)))?;
        if self.values.len() != expected_values {
            return Err(FeaturePacketError(format!(
                "{} value buffer length mismatch",
                self.spec.id
            )));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(FeaturePacketError(format!(
                "{} contains a non-finite value",
                self.spec.id
            )));
        }
        if self.available.len() != batch
            || self.status.len() != batch
            || self.reason.len() != batch
            || self.provenance.len() != batch
        {
            return Err(FeaturePacketError(format!(
                "{} row metadata length mismatch",
                self.spec.id
            )));
        }
        if self
            .provenance
            .iter()
            .flatten()
            .any(|item| item.trim().is_empty())
        {
            return Err(FeaturePacketError(format!(
                "{} contains empty provenance metadata",
                self.spec.id
            )));
        }
        if self.available.iter().any(|mask| *mask > 1) {
            return Err(FeaturePacketError(format!(
                "{} availability mask is not boolean",
                self.spec.id
            )));
        }
        for row in 0..batch {
            if self.available[row] == 1 {
                if self.status[row] != FeatureStatus::Computed {
                    return Err(FeaturePacketError(format!(
                        "{} available row has noncomputed status",
                        self.spec.id
                    )));
                }
            } else if self.status[row] == FeatureStatus::Computed
                || self.reason[row].trim().is_empty()
            {
                return Err(FeaturePacketError(format!(
                    "{} unavailable row lacks an explicit reason",
                    self.spec.id
                )));
            }
        }
        if let Some(mask) = &self.token_valid {
            if mask.iter().any(|value| *value > 1) {
                return Err(FeaturePacketError(format!(
                    "{} token mask is not boolean",
                    self.spec.id
                )));
            }
        }
        if let Some(incidence) = &self.incidence {
            if incidence
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(FeaturePacketError(format!(
                    "{} incidence must be finite and nonnegative",
                    self.spec.id
                )));
            }
        }
        Ok(())
    }

    pub fn row_is_available(&self, row: usize) -> bool {
        self.available.get(row).copied() == Some(1)
    }
}

#[derive(Clone, Debug)]
pub struct FeaturePacket {
    pub entries: Vec<FeatureEntry>,
    pub state_keys: Vec<StateIdentity>,
    pub alphabet_ids: Vec<String>,
    pub batch: usize,
    pub nodes: usize,
    pub valid_nodes: Vec<u8>,
    pub schema_id: String,
    pub policy_id: String,
    pub feature_id_manifest_id: &'static str,
}

impl FeaturePacket {
    pub fn new(
        entries: Vec<FeatureEntry>,
        states: &[MatrixState],
        nodes: usize,
        valid_nodes: Vec<u8>,
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
    ) -> Result<Self, FeaturePacketError> {
        let batch = states.len();
        if batch == 0 || nodes == 0 {
            return Err(FeaturePacketError(
                "feature packet needs a nonempty batch and node width".into(),
            ));
        }
        if entries.len() != FEATURE_COUNT {
            return Err(FeaturePacketError(format!(
                "expected exactly {FEATURE_COUNT} features"
            )));
        }
        if valid_nodes.len() != batch.checked_mul(nodes).unwrap_or(usize::MAX)
            || valid_nodes.iter().any(|value| *value > 1)
        {
            return Err(FeaturePacketError(
                "node validity mask shape/value mismatch".into(),
            ));
        }
        for (row, state) in states.iter().enumerate() {
            if state.dimension() as usize > nodes {
                return Err(FeaturePacketError(format!(
                    "state row {row} exceeds node width"
                )));
            }
            for node in 0..nodes {
                let expected = u8::from(node < state.dimension() as usize);
                if valid_nodes[row * nodes + node] != expected {
                    return Err(FeaturePacketError(format!(
                        "state row {row} has a noncanonical node mask"
                    )));
                }
            }
        }
        for (index, entry) in entries.iter().enumerate() {
            let expected = FeatureId::from_index(index)
                .ok_or_else(|| FeaturePacketError("feature manifest index overflow".into()))?;
            if entry.spec.id != expected {
                return Err(FeaturePacketError(format!(
                    "feature order mismatch at index {index}"
                )));
            }
            entry.validate(batch, nodes)?;
        }
        let schema_id = schema_id.into();
        let policy_id = policy_id.into();
        if schema_id.is_empty() || policy_id.is_empty() {
            return Err(FeaturePacketError(
                "schema and policy identities are required".into(),
            ));
        }
        Ok(Self {
            entries,
            state_keys: states.iter().map(StateIdentity::from).collect(),
            alphabet_ids: states
                .iter()
                .map(|state| state.alphabet_id().to_owned())
                .collect(),
            batch,
            nodes,
            valid_nodes,
            schema_id,
            policy_id,
            feature_id_manifest_id: FEATURE_ID_MANIFEST_ID,
        })
    }

    pub fn complete_103(&self) -> bool {
        self.entries
            .iter()
            .all(|entry| entry.available.iter().all(|value| *value == 1))
    }

    pub fn require_complete_103(&self) -> Result<(), FeaturePacketError> {
        let missing: Vec<String> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let rows: Vec<String> = entry
                    .available
                    .iter()
                    .enumerate()
                    .filter_map(|(row, value)| {
                        (*value == 0).then(|| {
                            format!(
                                "row {row}: {} ({})",
                                entry.status[row].as_str(),
                                entry.reason[row]
                            )
                        })
                    })
                    .collect();
                (!rows.is_empty()).then(|| format!("{}: {}", entry.spec.id, rows.join(", ")))
            })
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(FeaturePacketError(format!(
                "complete-103 rejected; {}",
                missing.join("; ")
            )))
        }
    }

    /// Copy one immutable batch row while preserving its feature masks and provenance.
    pub fn single_row(&self, row: usize) -> Result<Self, FeaturePacketError> {
        if self.batch == 0 || row >= self.batch {
            return Err(FeaturePacketError(
                "feature row is outside the packet".into(),
            ));
        }
        if self.state_keys.len() != self.batch
            || self.alphabet_ids.len() != self.batch
            || self.valid_nodes.len() != self.batch.checked_mul(self.nodes).unwrap_or(usize::MAX)
        {
            return Err(FeaturePacketError(
                "feature packet batch metadata is malformed".into(),
            ));
        }

        let mut entries = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if entry.shape.first().copied() != Some(self.batch)
                || entry.available.len() != self.batch
                || entry.status.len() != self.batch
                || entry.reason.len() != self.batch
                || entry.provenance.len() != self.batch
                || entry.values.len() % self.batch != 0
            {
                return Err(FeaturePacketError(format!(
                    "{} cannot be sliced from a malformed batch entry",
                    entry.spec.id
                )));
            }
            let values_per_row = entry.values.len() / self.batch;
            let values_start = row
                .checked_mul(values_per_row)
                .ok_or_else(|| FeaturePacketError("feature row offset overflow".into()))?;
            let values_end = values_start
                .checked_add(values_per_row)
                .ok_or_else(|| FeaturePacketError("feature row end overflow".into()))?;
            let mut shape = entry.shape.clone();
            shape[0] = 1;
            let sliced = FeatureEntry {
                spec: entry.spec.clone(),
                shape,
                values: entry
                    .values
                    .get(values_start..values_end)
                    .ok_or_else(|| FeaturePacketError("feature row is truncated".into()))?
                    .to_vec(),
                available: vec![entry.available[row]],
                status: vec![entry.status[row]],
                reason: vec![entry.reason[row].clone()],
                provenance: vec![entry.provenance[row].clone()],
                wire_roles: entry.wire_roles.clone(),
                token_valid: slice_optional_row(&entry.token_valid, self.batch, row)?,
                incidence: slice_optional_row(&entry.incidence, self.batch, row)?,
            };
            sliced.validate(1, self.nodes)?;
            entries.push(sliced);
        }

        let nodes_start = row
            .checked_mul(self.nodes)
            .ok_or_else(|| FeaturePacketError("node-mask row offset overflow".into()))?;
        let nodes_end = nodes_start
            .checked_add(self.nodes)
            .ok_or_else(|| FeaturePacketError("node-mask row end overflow".into()))?;
        Ok(Self {
            entries,
            state_keys: vec![self.state_keys[row].clone()],
            alphabet_ids: vec![self.alphabet_ids[row].clone()],
            batch: 1,
            nodes: self.nodes,
            valid_nodes: self
                .valid_nodes
                .get(nodes_start..nodes_end)
                .ok_or_else(|| FeaturePacketError("node-mask row is truncated".into()))?
                .to_vec(),
            schema_id: self.schema_id.clone(),
            policy_id: self.policy_id.clone(),
            feature_id_manifest_id: self.feature_id_manifest_id,
        })
    }

    pub fn feature(&self, id: FeatureId) -> &FeatureEntry {
        &self.entries[id.index()]
    }
}

fn slice_optional_row<T: Clone>(
    values: &Option<Vec<T>>,
    batch: usize,
    row: usize,
) -> Result<Option<Vec<T>>, FeaturePacketError> {
    match values {
        None => Ok(None),
        Some(values) if values.len() % batch == 0 => {
            let stride = values.len() / batch;
            let start = row
                .checked_mul(stride)
                .ok_or_else(|| FeaturePacketError("group row offset overflow".into()))?;
            let end = start
                .checked_add(stride)
                .ok_or_else(|| FeaturePacketError("group row end overflow".into()))?;
            values
                .get(start..end)
                .map(|slice| Some(slice.to_vec()))
                .ok_or_else(|| FeaturePacketError("group metadata is truncated".into()))
        }
        Some(_) => Err(FeaturePacketError(
            "group metadata batch length differs".into(),
        )),
    }
}
