use super::{FeatureEntry, FeaturePacket, FeaturePacketError, FeatureStatus, StateIdentity};
use crate::schema::{
    all_model_input_specs, FeatureId, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID,
    H07_POST_FORWARD_DIAGNOSTIC_ID, MODEL_INPUT_COUNT,
};
use crate::state::MatrixState;
use std::collections::HashSet;

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

    /// A packet has complete model-input coverage when all 102 inputs have
    /// explicit producer results for every row. Inapplicable results remain
    /// masked; H07 is checked separately as a post-forward diagnostic.
    pub fn has_complete_model_input_coverage(&self) -> bool {
        self.require_complete_model_input_coverage().is_ok()
    }

    pub fn require_complete_model_input_coverage(&self) -> Result<(), FeaturePacketError> {
        if self.entries.len() != FEATURE_COUNT
            || self.feature_id_manifest_id != FEATURE_ID_MANIFEST_ID
        {
            return Err(FeaturePacketError(
                "packet does not match the canonical 103-ID feature manifest".into(),
            ));
        }

        let expected: HashSet<_> = all_model_input_specs()
            .into_iter()
            .map(|spec| spec.id)
            .collect();
        let actual: HashSet<_> = self
            .entries
            .iter()
            .map(|entry| entry.spec.id)
            .filter(|id| *id != H07_POST_FORWARD_DIAGNOSTIC_ID)
            .collect();
        if expected.len() != MODEL_INPUT_COUNT || actual != expected {
            return Err(FeaturePacketError(format!(
                "packet must contain exactly {MODEL_INPUT_COUNT} canonical model inputs"
            )));
        }

        let mut failures = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            let expected_id = FeatureId::from_index(index)
                .ok_or_else(|| FeaturePacketError("feature manifest index overflow".into()))?;
            if entry.spec.id != expected_id {
                return Err(FeaturePacketError(format!(
                    "feature order mismatch at index {index}"
                )));
            }
            entry.validate(self.batch, self.nodes)?;

            if entry.spec.id == H07_POST_FORWARD_DIAGNOSTIC_ID {
                if entry
                    .status
                    .iter()
                    .any(|status| *status != FeatureStatus::NotRequested)
                    || entry.available.iter().any(|value| *value != 0)
                {
                    failures
                        .push("H07: post-forward diagnostic was included as a model input".into());
                }
                continue;
            }

            for row in 0..self.batch {
                let valid = match entry.status[row] {
                    FeatureStatus::Computed => entry.available[row] == 1,
                    FeatureStatus::Inapplicable => entry.available[row] == 0,
                    FeatureStatus::Unsupported
                    | FeatureStatus::ResourceLimit
                    | FeatureStatus::NotRequested => false,
                };
                if !valid {
                    failures.push(format!(
                        "{}: row {row}: {} ({})",
                        entry.spec.id,
                        entry.status[row].as_str(),
                        entry.reason[row]
                    ));
                }
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(FeaturePacketError(format!(
                "model-input coverage rejected; {}",
                failures.join("; ")
            )))
        }
    }
}
