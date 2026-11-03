//! Source-aligned full-graph objective and exposure accounting.
//!
//! Training records are validated as one batch before labels, feature forwards,
//! or gradients are admitted. Exact labels require a verified closed catalog;
//! verified reduction words are upper bounds only.

use crate::alphabet::CnotAlphabet;
use crate::bellman::{
    recompute_backup, select_backup_with_model_input_coverage, BellmanSelectionConfig,
};
use crate::checkpoint::CheckpointStore;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::FusionDistanceField;
use crate::nn_training::{NeuralTrainer, PreparedTrainingLoss, TchTrainingError};
use crate::packet::StateIdentity;
use crate::schema::all_model_input_specs;
use crate::state::MatrixState;
use crate::training_data::{elementary_lower_bound, ExactCatalog, TrainingRecord};
use std::collections::{HashMap, HashSet};
use tch::nn::VarStore;
use tch::{Kind, Reduction, Tensor};

mod exposure;
mod fingerprint;
mod losses;
mod runner;

pub use exposure::{ExposureAudit, ObjectiveDiagnostics, StateAdmission};
use exposure::{ExposureLedger, EXPOSURE_SCHEMA};
use fingerprint::{
    is_sha256_hex, objective_config_digest, training_records_digest, validate_identity,
};
use losses::{qhat_loss, teacher_loss};
pub use runner::FullGraphObjectiveRunner;

#[derive(Clone, Debug)]
pub struct FullGraphObjectiveConfig {
    pub bellman_weight: f64,
    pub supervision_weight: f64,
    pub qhat_weight: f64,
    pub teacher_weight: f64,
    pub neighbor_chunk_size: usize,
    pub teacher_temperature: f64,
    /// Must match the runner's sealed-data admission callback and is bound into
    /// the checkpoint contract, as in the reference objective.
    pub external_admission: bool,
    /// Stable SHA-256 identity of the exact admission policy/data boundary.
    /// A boolean alone cannot bind a resumed run to the same held-out policy.
    pub external_admission_sha256: Option<String>,
    /// Require all executable inputs from the 103-ID manifest; H07 is diagnostic-only.
    pub require_all_model_inputs: bool,
    pub compact_exposures: bool,
}

impl Default for FullGraphObjectiveConfig {
    fn default() -> Self {
        Self {
            bellman_weight: 1.0,
            supervision_weight: 1.0,
            qhat_weight: 0.1,
            teacher_weight: 0.0,
            neighbor_chunk_size: 32,
            teacher_temperature: 4.0,
            external_admission: false,
            external_admission_sha256: None,
            require_all_model_inputs: true,
            compact_exposures: false,
        }
    }
}

impl FullGraphObjectiveConfig {
    fn validate(&self) -> Result<(), String> {
        let weights = [
            self.bellman_weight,
            self.supervision_weight,
            self.qhat_weight,
            self.teacher_weight,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || weights.iter().sum::<f64>() <= 0.0
        {
            return Err("objective weights must be finite, nonnegative, and not all zero".into());
        }
        if self.neighbor_chunk_size == 0 {
            return Err("positive neighbor chunk size required".into());
        }
        if !self.teacher_temperature.is_finite() || self.teacher_temperature <= 0.0 {
            return Err("positive finite teacher temperature required".into());
        }
        match (
            self.external_admission,
            self.external_admission_sha256.as_deref(),
        ) {
            (false, None) => {}
            (true, Some(digest)) if is_sha256_hex(digest) => {}
            (false, Some(_)) => {
                return Err("admission fingerprint supplied without external admission".into())
            }
            (true, None) => {
                return Err("external admission requires a stable SHA-256 fingerprint".into())
            }
            (true, Some(_)) => {
                return Err("external admission fingerprint must be lowercase SHA-256 hex".into())
            }
        }
        Ok(())
    }
}

pub struct FullGraphObjective {
    alphabets: HashMap<String, CnotAlphabet>,
    exact_catalogs: HashMap<String, ExactCatalog>,
    feature_contract_sha256: String,
    training_records_sha256: Option<String>,
    frozen_dataset_sha256: Option<String>,
    policy: FeaturePolicy,
    forbidden: HashSet<StateIdentity>,
    config: FullGraphObjectiveConfig,
    config_sha256: String,
    exposures: ExposureLedger,
    last_diagnostics: ObjectiveDiagnostics,
}

impl FullGraphObjective {
    pub fn new(
        alphabets: Vec<CnotAlphabet>,
        registry: &FeatureRegistry,
        policy: FeaturePolicy,
        exact_catalogs: Vec<ExactCatalog>,
        forbidden_states: Vec<MatrixState>,
        config: FullGraphObjectiveConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let policy_id = policy.id().map_err(|error| error.to_string())?;
        if config.require_all_model_inputs {
            let selected: HashSet<_> = policy.enabled_ids.iter().copied().collect();
            let expected: HashSet<_> = all_model_input_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect();
            if selected != expected {
                return Err(
                    "full graph training requires every executable model input from the canonical 103-ID manifest".into(),
                );
            }
        }

        let mut alphabet_map = HashMap::with_capacity(alphabets.len());
        for alphabet in alphabets {
            if alphabet_map
                .insert(alphabet.id().to_owned(), alphabet)
                .is_some()
            {
                return Err("duplicate alphabet identity".into());
            }
        }
        if alphabet_map.is_empty() {
            return Err("full graph objective requires at least one alphabet".into());
        }

        let mut exact_map = HashMap::with_capacity(exact_catalogs.len());
        for catalog in exact_catalogs {
            if !alphabet_map.contains_key(catalog.alphabet_id()) {
                return Err(
                    "exact catalog uses an alphabet not registered by the objective".into(),
                );
            }
            let key = catalog.alphabet_id().to_owned();
            if exact_map.insert(key, catalog).is_some() {
                return Err("duplicate exact catalog for one alphabet".into());
            }
        }

        let mut forbidden = HashSet::with_capacity(forbidden_states.len());
        for state in forbidden_states {
            let alphabet = alphabet_map
                .get(state.alphabet_id())
                .ok_or_else(|| "forbidden state uses an unregistered alphabet".to_owned())?;
            if state.dimension() != alphabet.dimension() {
                return Err("forbidden state dimension differs from its alphabet".into());
            }
            forbidden.insert(StateIdentity::from(&state));
        }

        let feature_contract_sha256 = registry.contract_sha256();
        let config_sha256 = objective_config_digest(
            &alphabet_map,
            &exact_map,
            &policy_id,
            &feature_contract_sha256,
            &forbidden,
            &config,
            None,
            None,
        );
        Ok(Self {
            alphabets: alphabet_map,
            exact_catalogs: exact_map,
            feature_contract_sha256,
            training_records_sha256: None,
            frozen_dataset_sha256: None,
            policy,
            forbidden,
            config,
            config_sha256,
            exposures: ExposureLedger::default(),
            last_diagnostics: ObjectiveDiagnostics::default(),
        })
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn diagnostics(&self) -> &ObjectiveDiagnostics {
        &self.last_diagnostics
    }

    pub fn audit(&self) -> ExposureAudit {
        self.exposures
            .audit(&self.forbidden, self.config.compact_exposures)
    }

    pub fn state_dict(&self) -> serde_json::Value {
        let mut state = self.exposures.state_dict(self.config.compact_exposures);
        state["config_sha256"] = serde_json::Value::String(self.config_sha256.clone());
        state
    }

    /// Stable checkpoint contract for `Checkpoint.contract`.
    pub fn checkpoint_contract(&self) -> serde_json::Value {
        serde_json::json!({
            "id": "full_graph_reference_v3",
            "objective_sha256": self.config_sha256,
            "feature_registry_sha256": self.feature_contract_sha256,
            "frozen_dataset_sha256": self.frozen_dataset_sha256,
            "training_records_sha256": self.training_records_sha256,
            "exposure_schema": EXPOSURE_SCHEMA,
        })
    }

    /// Bind only the artifact identity before constructing a runner.
    /// Prefer `bind_frozen_dataset` when the loaded dataset is available so its
    /// train-panel digest is bound at the same time.
    pub fn bind_frozen_dataset_sha256(&mut self, sha256: &str) -> Result<(), String> {
        if !is_sha256_hex(sha256) {
            return Err("frozen dataset fingerprint must be lowercase SHA-256 hex".into());
        }
        if let Some(bound) = &self.frozen_dataset_sha256 {
            return if bound == sha256 {
                Ok(())
            } else {
                Err("objective is already bound to a different frozen dataset".into())
            };
        }
        if self.training_records_sha256.is_some() {
            return Err("frozen dataset must be bound before constructing the runner".into());
        }
        let policy_id = self.policy.id().map_err(|error| error.to_string())?;
        self.config_sha256 = objective_config_digest(
            &self.alphabets,
            &self.exact_catalogs,
            &policy_id,
            &self.feature_contract_sha256,
            &self.forbidden,
            &self.config,
            None,
            Some(sha256),
        );
        self.frozen_dataset_sha256 = Some(sha256.to_owned());
        Ok(())
    }

    /// Bind the dataset artifact and the exact training records obtained from
    /// its sealed train panel. Runner construction rechecks this record digest.
    pub fn bind_frozen_dataset(
        &mut self,
        dataset: &crate::frozen_data::FrozenDataset,
    ) -> Result<(), String> {
        if !is_sha256_hex(&dataset.sha256) {
            return Err("frozen dataset fingerprint must be lowercase SHA-256 hex".into());
        }
        if let Some(bound) = &self.frozen_dataset_sha256 {
            if bound != &dataset.sha256 {
                return Err("objective is already bound to a different frozen dataset".into());
            }
        } else if self.training_records_sha256.is_some() {
            return Err("frozen dataset must be bound before constructing the runner".into());
        }
        let records_digest = training_records_digest(dataset.train.records());
        if let Some(bound) = &self.training_records_sha256 {
            if bound != &records_digest {
                return Err("frozen dataset train panel differs from the bound record set".into());
            }
        }
        let policy_id = self.policy.id().map_err(|error| error.to_string())?;
        self.config_sha256 = objective_config_digest(
            &self.alphabets,
            &self.exact_catalogs,
            &policy_id,
            &self.feature_contract_sha256,
            &self.forbidden,
            &self.config,
            Some(&records_digest),
            Some(&dataset.sha256),
        );
        self.training_records_sha256 = Some(records_digest);
        self.frozen_dataset_sha256 = Some(dataset.sha256.clone());
        Ok(())
    }

    fn bind_training_records(&mut self, records: &[TrainingRecord]) -> Result<(), String> {
        let digest = training_records_digest(records);
        if let Some(bound) = &self.training_records_sha256 {
            if bound != &digest {
                return Err(
                    "runner training records differ from the objective checkpoint contract".into(),
                );
            }
            return Ok(());
        }

        let policy_id = self.policy.id().map_err(|error| error.to_string())?;
        self.config_sha256 = objective_config_digest(
            &self.alphabets,
            &self.exact_catalogs,
            &policy_id,
            &self.feature_contract_sha256,
            &self.forbidden,
            &self.config,
            Some(&digest),
            self.frozen_dataset_sha256.as_deref(),
        );
        self.training_records_sha256 = Some(digest);
        Ok(())
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

    fn compute_loss(
        &mut self,
        model: &FusionDistanceField,
        variables: &VarStore,
        registry: &FeatureRegistry,
        records: &[TrainingRecord],
        admission: Option<&dyn StateAdmission>,
    ) -> Result<PreparedTrainingLoss, String> {
        if records.is_empty() {
            return Err("empty training batch".into());
        }
        if self.config.compact_exposures && admission.is_none() {
            return Err("compact exposure audit requires external admission".into());
        }
        if model.device() != variables.device() {
            return Err("model and variable store are on different devices".into());
        }
        if self.config.external_admission != admission.is_some() {
            return Err("runner external admission differs from the objective contract".into());
        }
        if admission.map(StateAdmission::contract_sha256)
            != self.config.external_admission_sha256.as_deref()
        {
            return Err("runner admission fingerprint differs from the objective contract".into());
        }

        struct Validated<'a> {
            record: &'a TrainingRecord,
            alphabet: &'a CnotAlphabet,
            lower_bound: u64,
            upper_bound: u64,
            successors: Vec<MatrixState>,
        }

        // Finish all split, certificate, bound, forbidden-neighbor, and
        // external-admission checks before mutating exposures or calling the model.
        let mut validated = Vec::with_capacity(records.len());
        for record in records {
            let alphabet = self
                .alphabets
                .get(record.state.alphabet_id())
                .ok_or_else(|| "training record uses an unregistered alphabet".to_owned())?;
            let exact = self.exact_catalogs.get(alphabet.id());
            let upper_bound = record
                .validate_for_training(alphabet, exact)
                .map_err(|error| error.to_string())?;
            let lower_bound =
                elementary_lower_bound(&record.state).map_err(|error| error.to_string())?;
            if lower_bound > upper_bound {
                return Err("verified word upper bound is below the source lower bound".into());
            }

            let mut successors = Vec::with_capacity(alphabet.gates().len());
            for gate in alphabet.gates().iter().copied() {
                let mut successor = record.state.clone();
                successor
                    .apply_cnot(gate)
                    .map_err(|error| error.to_string())?;
                if self.forbidden.contains(&StateIdentity::from(&successor)) {
                    return Err("forbidden state in complete backup; reject whole parent".into());
                }
                if let Some(admission) = admission {
                    admission.admit(&successor)?;
                }
                successors.push(successor);
            }
            if self.forbidden.contains(&StateIdentity::from(&record.state)) {
                return Err("forbidden state in training batch".into());
            }
            if let Some(admission) = admission {
                admission.admit(&record.state)?;
            }
            validated.push(Validated {
                record,
                alphabet,
                lower_bound,
                upper_bound,
                successors,
            });
        }

        // Feature packets are built once per example and retained only through
        // that example's loss construction. This keeps candidate coverage
        // fail-closed without multiplying provider work or holding every
        // Bellman packet in the batch at once.
        let mut terms = Vec::with_capacity(validated.len());
        let mut gradient_states = Vec::with_capacity(validated.len() * 2);
        let mut diagnostics = ObjectiveDiagnostics {
            examples: validated.len(),
            ..ObjectiveDiagnostics::default()
        };
        for item in &validated {
            let state = &item.record.state;
            // Terminal Bellman rows have no selected successor, but terminal
            // Q-hat still targets every primitive action. Check their packets
            // before the loss pass, using bounded per-item memory.
            if self.config.require_all_model_inputs
                && self.config.qhat_weight > 0.0
                && state.is_identity()
            {
                let packet = registry
                    .build(&item.successors, &self.policy)
                    .map_err(|error| error.to_string())?;
                packet
                    .require_complete_model_input_coverage()
                    .map_err(|error| error.to_string())?;
            }
            let selection = select_backup_with_model_input_coverage(
                model,
                variables,
                std::slice::from_ref(state),
                registry,
                &self.policy,
                item.alphabet,
                BellmanSelectionConfig {
                    chunk_size: Some(self.config.neighbor_chunk_size),
                    require_complete_model_input_coverage: self.config.require_all_model_inputs,
                },
            )
            .map_err(|error| error.to_string())?;
            let branch = selection
                .branches()
                .next()
                .ok_or_else(|| "Bellman selection did not return a branch".to_owned())?;
            diagnostics.primitive_candidates = diagnostics
                .primitive_candidates
                .checked_add(branch.primitive_count)
                .ok_or_else(|| "primitive candidate counter overflow".to_owned())?;
            let bellman = recompute_backup(model, variables, &selection)
                .map_err(|error| error.to_string())?;

            let parent_packet = selection
                .parent_packet(0)
                .ok_or_else(|| "Bellman selection did not retain its parent packet".to_owned())?;
            let output = model
                .forward(parent_packet)
                .map_err(|error| error.to_string())?;
            if output.value.numel() != 1 || output.value.isfinite().all().int64_value(&[]) == 0 {
                return Err("objective parent value is not a finite scalar".into());
            }
            let value = output.value.select(0, 0);
            let lower = Tensor::from_slice(&[item.lower_bound as f64])
                .to_device(value.device())
                .to_kind(value.kind())
                .select(0, 0);
            let upper = Tensor::from_slice(&[item.upper_bound as f64])
                .to_device(value.device())
                .to_kind(value.kind())
                .select(0, 0);
            let interval = (lower - &value).relu().pow_tensor_scalar(2.0)
                + (&value - upper).relu().pow_tensor_scalar(2.0);
            let exact_loss = if let Some(exact_distance) = item.record.exact_distance {
                diagnostics.exact_examples += 1;
                let target = Tensor::from_slice(&[exact_distance as f64])
                    .to_device(value.device())
                    .to_kind(value.kind())
                    .select(0, 0);
                value.smooth_l1_loss(&target, Reduction::Mean, 1.0)
            } else {
                value.shallow_clone() * 0.0
            };

            let qhat_loss = if self.config.qhat_weight > 0.0 {
                qhat_loss(
                    model,
                    registry,
                    &self.policy,
                    item.alphabet,
                    state,
                    &output,
                    branch,
                )?
            } else {
                output
                    .qhat
                    .masked_fill(&Tensor::ones_like(&output.qhat).to_kind(Kind::Bool), 0.0)
                    .sum(output.qhat.kind())
                    * 0.0
            };
            let teacher = if self.config.teacher_weight > 0.0 {
                let (loss, used) = teacher_loss(
                    &output.proposal_logits,
                    &output.action_mask,
                    item.record,
                    item.alphabet,
                    self.config.teacher_temperature,
                )?;
                diagnostics.teacher_examples += usize::from(used);
                loss
            } else {
                output
                    .proposal_logits
                    .masked_fill(&output.action_mask.logical_not(), 0.0)
                    .sum(output.proposal_logits.kind())
                    * 0.0
            };

            let term = bellman.loss * self.config.bellman_weight
                + (interval + exact_loss) * self.config.supervision_weight
                + qhat_loss * self.config.qhat_weight
                + teacher * self.config.teacher_weight;
            if term.requires_grad() {
                gradient_states.extend(selection.gradient_exposure_states());
            }
            terms.push(term);
        }
        if terms.is_empty() {
            return Err("empty full graph objective".into());
        }
        let loss_kind = terms[0].kind();
        let loss = Tensor::stack(&terms, 0).mean(loss_kind);
        if loss.numel() != 1 || loss.isfinite().all().int64_value(&[]) == 0 {
            return Err("nonfinite full graph objective loss".into());
        }
        // Commit exposure changes only after the complete batch has produced a
        // finite loss. A provider/model failure cannot leave a partial audit.
        for item in &validated {
            if self.config.supervision_weight > 0.0 {
                self.exposures
                    .observe(self.config.compact_exposures, 0, &item.record.state);
            }
            self.exposures
                .observe(self.config.compact_exposures, 1, &item.record.state);
            if !item.record.state.is_identity() || self.config.qhat_weight > 0.0 {
                for successor in &item.successors {
                    self.exposures
                        .observe(self.config.compact_exposures, 1, successor);
                }
            }
        }
        self.last_diagnostics = diagnostics;
        Ok(PreparedTrainingLoss {
            loss,
            gradient_states,
        })
    }
}
