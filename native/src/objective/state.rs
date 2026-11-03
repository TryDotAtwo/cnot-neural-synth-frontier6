//! Exposure-ledger serialization and checkpoint restoration for the objective.
use super::{EXPOSURE_SCHEMA, ExposureLedger, FullGraphObjective, validate_identity};
use crate::checkpoint::CheckpointStore;
use crate::nn_training::{NeuralTrainer, TchTrainingError};
use crate::packet::StateIdentity;
use crate::training_data::TrainingRecord;
use std::collections::HashSet;
use tch::nn::VarStore;

impl FullGraphObjective {
    pub fn state_dict(&self) -> serde_json::Value {
        let mut state = self.exposures.state_dict(self.config.compact_exposures);
        state["config_sha256"] = serde_json::Value::String(self.config_sha256.clone());
        state
    }

    /// Serialize exposure state for `NeuralTrainer::set_objective_state`.
    pub fn checkpoint_state_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&self.state_dict())
            .map_err(|error| format!("could not encode objective checkpoint state: {error}"))
    }

    /// Restore exposure state from `NeuralTrainer::objective_state`.
    pub fn restore_checkpoint_state(&mut self, bytes: &[u8]) -> Result<(), String> {
        let state: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("objective checkpoint state is invalid JSON: {error}"))?;
        self.load_state_dict(&state)
    }

    pub fn load_state_dict(&mut self, state: &serde_json::Value) -> Result<(), String> {
        let proposed = self.decode_state_dict(state)?;
        self.exposures = proposed;
        Ok(())
    }

    fn decode_state_dict(&self, state: &serde_json::Value) -> Result<ExposureLedger, String> {
        let object = state
            .as_object()
            .ok_or_else(|| "objective exposure state must be a JSON object".to_owned())?;
        if object.get("schema").and_then(serde_json::Value::as_str) != Some(EXPOSURE_SCHEMA) {
            return Err("objective exposure checkpoint schema differs".into());
        }
        if object
            .get("config_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(self.config_sha256.as_str())
        {
            return Err("objective exposure checkpoint contract differs".into());
        }
        let mode = if self.config.compact_exposures {
            "compact"
        } else {
            "full"
        };
        if object.get("mode").and_then(serde_json::Value::as_str) != Some(mode) {
            return Err("objective exposure checkpoint mode differs".into());
        }

        if self.config.compact_exposures {
            if object.len() != 4 {
                return Err("compact objective exposure checkpoint has unexpected fields".into());
            }
            let counts = object
                .get("counts")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| "compact exposure counts are missing".to_owned())?;
            if counts.len() != 3 {
                return Err("compact exposure checkpoint needs three counters".into());
            }
            let read = |name: &str| {
                counts
                    .get(name)
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| format!("invalid compact exposure counter: {name}"))
            };
            return Ok(ExposureLedger {
                compact_counts: [read("label")?, read("forward")?, read("gradient")?],
                ..ExposureLedger::default()
            });
        }

        if object.len() != 6 {
            return Err("full objective exposure checkpoint has unexpected fields".into());
        }
        let parse_set = |name: &str| -> Result<HashSet<StateIdentity>, String> {
            let rows = object
                .get(name)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("objective exposure set is missing: {name}"))?;
            let mut result = HashSet::with_capacity(rows.len());
            for row in rows {
                let identity: StateIdentity = serde_json::from_value(row.clone())
                    .map_err(|error| format!("invalid exposure identity: {error}"))?;
                validate_identity(&identity, &self.alphabets)?;
                if !result.insert(identity) {
                    return Err(format!("duplicate state identity in exposure set: {name}"));
                }
            }
            Ok(result)
        };
        let labels = parse_set("seen_as_label")?;
        let forward = parse_set("seen_by_forward")?;
        let gradient = parse_set("seen_by_gradient")?;
        let proposed = ExposureLedger {
            seen_as_label: labels,
            seen_by_forward: forward,
            seen_by_gradient: gradient,
            compact_counts: [0; 3],
        };
        let audit = proposed.audit(&self.forbidden, false);
        if !audit.strict_unseen {
            return Err("restored objective exposures overlap the frozen forbidden set".into());
        }
        Ok(proposed)
    }

    fn decode_checkpoint_state_bytes(&self, bytes: &[u8]) -> Result<ExposureLedger, String> {
        let state: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("objective checkpoint state is invalid JSON: {error}"))?;
        self.decode_state_dict(&state)
    }

    /// Restore model, optimizer, sampler, objective ledger, and RNG bytes from
    /// one hash-validated checkpoint bound to this exact ordered record set.
    /// Objective payload validation happens before the trainer mutates tensors.
    pub fn restore_training_checkpoint(
        &mut self,
        trainer: &mut NeuralTrainer,
        variables: &mut VarStore,
        store: &CheckpointStore,
        records: &[TrainingRecord],
    ) -> Result<(), TchTrainingError> {
        if records.is_empty() {
            return Err(TchTrainingError::Objective(
                "training record set is empty".into(),
            ));
        }
        self.bind_training_records(records)
            .map_err(TchTrainingError::Objective)?;
        let contract = self.checkpoint_contract();
        let checkpoint = store.load_latest(&contract)?;
        let exposures = self
            .decode_checkpoint_state_bytes(&checkpoint.objective_state)
            .map_err(TchTrainingError::Objective)?;
        trainer.restore_checkpoint_payload(variables, checkpoint, &contract)?;
        self.exposures = exposures;
        Ok(())
    }
}
