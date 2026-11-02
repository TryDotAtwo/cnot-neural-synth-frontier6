//! Source-aligned full-graph objective and exposure accounting.
//!
//! Training records are validated as one batch before labels, feature forwards,
//! or gradients are admitted. Exact labels require a verified closed catalog;
//! verified reduction words are upper bounds only.

use crate::alphabet::CnotAlphabet;
use crate::bellman::{recompute_backup, select_backup, BellmanBranchAudit};
use crate::checkpoint::{CheckpointManifest, CheckpointStore};
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::FusionDistanceField;
use crate::nn_training::{
    DifferentiableObjective, NeuralTrainer, PreparedTrainingLoss, TchTrainingError,
};
use crate::packet::StateIdentity;
use crate::schema::all_feature_specs;
use crate::state::{CnotGate, MatrixState};
use crate::training_data::{elementary_lower_bound, ExactCatalog, TrainingRecord};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use tch::nn::VarStore;
use tch::{Kind, Reduction, Tensor};

const EXPOSURE_SCHEMA: &str = "native_full_graph_exposures_v1";

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
    pub require_complete_103: bool,
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
            require_complete_103: true,
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

#[derive(Clone, Debug, Default)]
struct ExposureLedger {
    seen_as_label: HashSet<StateIdentity>,
    seen_by_forward: HashSet<StateIdentity>,
    seen_by_gradient: HashSet<StateIdentity>,
    compact_counts: [u64; 3],
}

impl ExposureLedger {
    fn observe(&mut self, compact: bool, kind: usize, state: &MatrixState) {
        if compact {
            self.compact_counts[kind] = self.compact_counts[kind].saturating_add(1);
        } else {
            let target = match kind {
                0 => &mut self.seen_as_label,
                1 => &mut self.seen_by_forward,
                _ => &mut self.seen_by_gradient,
            };
            target.insert(StateIdentity::from(state));
        }
    }

    fn audit(&self, forbidden: &HashSet<StateIdentity>, compact: bool) -> ExposureAudit {
        let label_overlap = self.seen_as_label.intersection(forbidden).count();
        let forward_overlap = self.seen_by_forward.intersection(forbidden).count();
        let gradient_overlap = self.seen_by_gradient.intersection(forbidden).count();
        let strict_unseen = label_overlap == 0 && forward_overlap == 0 && gradient_overlap == 0;
        let (label_count, forward_count, gradient_count) = if compact {
            (
                self.compact_counts[0] as usize,
                self.compact_counts[1] as usize,
                self.compact_counts[2] as usize,
            )
        } else {
            (
                self.seen_as_label.len(),
                self.seen_by_forward.len(),
                self.seen_by_gradient.len(),
            )
        };
        let set_sha256 = if compact {
            HashMap::new()
        } else {
            HashMap::from([
                ("labels".into(), identity_set_digest(&self.seen_as_label)),
                ("forward".into(), identity_set_digest(&self.seen_by_forward)),
                (
                    "gradient".into(),
                    identity_set_digest(&self.seen_by_gradient),
                ),
                ("forbidden".into(), identity_set_digest(forbidden)),
            ])
        };
        ExposureAudit {
            strict_unseen,
            audit_mode: if compact {
                "external_admission_before_forward"
            } else {
                "stored_full_state_identities"
            },
            label_count,
            forward_count,
            gradient_count,
            forbidden_count: forbidden.len(),
            label_overlap,
            forward_overlap,
            gradient_overlap,
            set_sha256,
        }
    }

    fn state_dict(&self, compact: bool) -> serde_json::Value {
        if compact {
            serde_json::json!({
                "schema": EXPOSURE_SCHEMA,
                "mode": "compact",
                "counts": {
                    "label": self.compact_counts[0],
                    "forward": self.compact_counts[1],
                    "gradient": self.compact_counts[2]
                }
            })
        } else {
            serde_json::json!({
                "schema": EXPOSURE_SCHEMA,
                "mode": "full",
                "seen_as_label": sorted_identities(&self.seen_as_label),
                "seen_by_forward": sorted_identities(&self.seen_by_forward),
                "seen_by_gradient": sorted_identities(&self.seen_by_gradient)
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposureAudit {
    pub strict_unseen: bool,
    pub audit_mode: &'static str,
    pub label_count: usize,
    pub forward_count: usize,
    pub gradient_count: usize,
    pub forbidden_count: usize,
    pub label_overlap: usize,
    pub forward_overlap: usize,
    pub gradient_overlap: usize,
    pub set_sha256: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectiveDiagnostics {
    pub examples: usize,
    pub primitive_candidates: usize,
    pub exact_examples: usize,
    pub teacher_examples: usize,
}

/// Checks a state against an external sealed-data/admission policy before any
/// feature forward. Implementations must fail closed for unknown identities.
pub trait StateAdmission {
    /// Stable identity of the admission implementation and the frozen split
    /// manifest it enforces. It is stored in the objective/checkpoint contract.
    fn contract_sha256(&self) -> &str;
    fn admit(&self, state: &MatrixState) -> Result<(), String>;
}

pub struct FullGraphObjective {
    alphabets: HashMap<String, CnotAlphabet>,
    exact_catalogs: HashMap<String, ExactCatalog>,
    feature_contract_sha256: String,
    training_records_sha256: Option<String>,
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
        if config.require_complete_103 {
            let selected: HashSet<_> = policy.enabled_ids.iter().copied().collect();
            let expected: HashSet<_> = all_feature_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect();
            if selected != expected {
                return Err(
                    "full graph training requires the canonical complete-103 feature policy".into(),
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
        );
        Ok(Self {
            alphabets: alphabet_map,
            exact_catalogs: exact_map,
            feature_contract_sha256,
            training_records_sha256: None,
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
            "id": "full_graph_reference_v1",
            "objective_sha256": self.config_sha256,
            "feature_registry_sha256": self.feature_contract_sha256,
            "training_records_sha256": self.training_records_sha256,
            "exposure_schema": EXPOSURE_SCHEMA,
        })
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

        // Fail closed on incomplete inputs. Reserved/unavailable features are
        // never silently treated as a complete 103-feature training row.
        if self.config.require_complete_103 {
            for item in &validated {
                let mut states = Vec::with_capacity(1 + item.successors.len());
                states.push(item.record.state.clone());
                if !item.record.state.is_identity() || self.config.qhat_weight > 0.0 {
                    states.extend(item.successors.iter().cloned());
                }
                let packet = registry
                    .build(&states, &self.policy)
                    .map_err(|error| error.to_string())?;
                packet
                    .require_complete_103()
                    .map_err(|error| error.to_string())?;
            }
        }

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

        let mut terms = Vec::with_capacity(validated.len());
        let mut gradient_states = Vec::with_capacity(validated.len() * 2);
        let mut diagnostics = ObjectiveDiagnostics {
            examples: validated.len(),
            ..ObjectiveDiagnostics::default()
        };
        for item in validated {
            let state = &item.record.state;
            let selection = select_backup(
                model,
                variables,
                std::slice::from_ref(state),
                registry,
                &self.policy,
                item.alphabet,
                Some(self.config.neighbor_chunk_size),
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

            let parent_packet = registry
                .build(std::slice::from_ref(state), &self.policy)
                .map_err(|error| error.to_string())?;
            if self.config.require_complete_103 {
                parent_packet
                    .require_complete_103()
                    .map_err(|error| error.to_string())?;
            }
            let output = model
                .forward(&parent_packet)
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
        self.last_diagnostics = diagnostics;
        Ok(PreparedTrainingLoss {
            loss,
            gradient_states,
        })
    }
}

pub struct FullGraphObjectiveRunner<'a> {
    objective: &'a mut FullGraphObjective,
    model: &'a FusionDistanceField,
    variables: &'a VarStore,
    registry: &'a FeatureRegistry,
    records: &'a [TrainingRecord],
    admission: Option<&'a dyn StateAdmission>,
}

impl<'a> FullGraphObjectiveRunner<'a> {
    pub fn new(
        objective: &'a mut FullGraphObjective,
        model: &'a FusionDistanceField,
        variables: &'a VarStore,
        registry: &'a FeatureRegistry,
        records: &'a [TrainingRecord],
        admission: Option<&'a dyn StateAdmission>,
    ) -> Result<Self, String> {
        if records.is_empty() {
            return Err("training record set is empty".into());
        }
        if objective.config.compact_exposures && admission.is_none() {
            return Err("compact exposure audit requires external admission".into());
        }
        if objective.config.external_admission != admission.is_some() {
            return Err("runner external admission differs from the objective contract".into());
        }
        if admission.map(StateAdmission::contract_sha256)
            != objective.config.external_admission_sha256.as_deref()
        {
            return Err("runner admission fingerprint differs from the objective contract".into());
        }
        if objective.feature_contract_sha256 != registry.contract_sha256() {
            return Err("runner feature registry differs from the objective contract".into());
        }
        registry
            .selected_specs(&objective.policy)
            .map_err(|error| format!("objective feature policy is not executable: {error}"))?;
        if model.device() != variables.device() {
            return Err("model and variable store are on different devices".into());
        }
        objective.bind_training_records(records)?;
        Ok(Self {
            objective,
            model,
            variables,
            registry,
            records,
            admission,
        })
    }
}

impl DifferentiableObjective for FullGraphObjectiveRunner<'_> {
    fn compute(&mut self, indices: &[usize]) -> Result<PreparedTrainingLoss, String> {
        if indices.is_empty() {
            return Err("training sampler returned an empty batch".into());
        }
        let records = indices
            .iter()
            .map(|index| {
                self.records
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| "training sampler returned an out-of-range record".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.objective.compute_loss(
            self.model,
            self.variables,
            self.registry,
            &records,
            self.admission,
        )
    }

    fn observe_gradient(&mut self, states: &[MatrixState]) {
        for state in states {
            self.objective
                .exposures
                .observe(self.objective.config.compact_exposures, 2, state);
        }
    }
}

impl FullGraphObjectiveRunner<'_> {
    /// Return the checkpoint identity after binding it to this ordered record
    /// set. A resume with changed data therefore fails contract validation.
    pub fn checkpoint_contract(&self) -> serde_json::Value {
        self.objective.checkpoint_contract()
    }

    /// Snapshot the objective state using the same record-bound contract.
    pub fn checkpoint_state_bytes(&self) -> Result<Vec<u8>, String> {
        self.objective.checkpoint_state_bytes()
    }

    /// Save the trainer and objective as one checkpoint using the runner's
    /// record-bound contract. Callers cannot accidentally save exposure state
    /// under a different data contract.
    pub fn save_checkpoint(
        &self,
        trainer: &mut NeuralTrainer,
        store: &CheckpointStore,
        update_best: bool,
    ) -> Result<CheckpointManifest, TchTrainingError> {
        let objective_state = self
            .checkpoint_state_bytes()
            .map_err(TchTrainingError::Objective)?;
        trainer.set_objective_state(objective_state);
        trainer.save_checkpoint(
            self.variables,
            store,
            self.objective.checkpoint_contract(),
            update_best,
        )
    }
}

fn qhat_loss(
    model: &FusionDistanceField,
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    state: &MatrixState,
    output: &crate::model::FieldOutput,
    branch: &BellmanBranchAudit,
) -> Result<Tensor, String> {
    let mut targets = Vec::new();
    if state.is_identity() {
        let mut successors = Vec::with_capacity(alphabet.gates().len());
        let mut costs = Vec::with_capacity(alphabet.gates().len());
        for (index, gate) in alphabet.gates().iter().copied().enumerate() {
            let mut successor = state.clone();
            successor
                .apply_cnot(gate)
                .map_err(|error| error.to_string())?;
            successors.push(successor);
            costs.push(f64::from(
                alphabet
                    .cost_at(index)
                    .ok_or_else(|| "alphabet gate cost is missing".to_owned())?,
            ));
        }
        if !successors.is_empty() {
            let packet = registry
                .build(&successors, policy)
                .map_err(|error| error.to_string())?;
            let values = tch::no_grad(|| model.forward(&packet))
                .map_err(|error| error.to_string())?
                .value;
            if values.numel() != successors.len() || values.isfinite().all().int64_value(&[]) == 0 {
                return Err("nonfinite terminal Q-hat target values".into());
            }
            for (index, gate) in alphabet.gates().iter().copied().enumerate() {
                targets.push((gate, costs[index] + values.double_value(&[index as i64])));
            }
        }
    } else {
        if branch.primitive_totals.len() != alphabet.gates().len() {
            return Err(
                "complete Bellman selection did not retain every primitive Q-hat target".into(),
            );
        }
        targets.extend(
            alphabet
                .gates()
                .iter()
                .copied()
                .zip(branch.primitive_totals.iter().copied()),
        );
    }
    masked_smooth_l1(&output.qhat, &output.action_mask, &targets)
}

fn masked_smooth_l1(
    predictions: &Tensor,
    action_mask: &Tensor,
    targets: &[(CnotGate, f64)],
) -> Result<Tensor, String> {
    if predictions.size() != action_mask.size() || predictions.size().first() != Some(&1) {
        return Err("Q-hat/action-mask tensor shape mismatch".into());
    }
    if targets.is_empty() {
        return Ok(predictions
            .masked_fill(&Tensor::ones_like(predictions).to_kind(Kind::Bool), 0.0)
            .sum(predictions.kind())
            * 0.0);
    }
    let nodes = usize::try_from(predictions.size()[1])
        .map_err(|_| "Q-hat node dimension exceeds host index range")?;
    let mut indices = Vec::with_capacity(targets.len());
    let mut values = Vec::with_capacity(targets.len());
    for (gate, target) in targets {
        if !target.is_finite() || gate.control as usize >= nodes || gate.target as usize >= nodes {
            return Err("invalid Q-hat target or gate index".into());
        }
        let index = usize::try_from(gate.control)
            .ok()
            .and_then(|control| control.checked_mul(nodes))
            .and_then(|offset| offset.checked_add(gate.target as usize))
            .ok_or_else(|| "Q-hat action index overflow".to_owned())?;
        indices.push(i64::try_from(index).map_err(|_| "Q-hat index exceeds tensor range")?);
        values.push(*target);
    }
    let indices = Tensor::from_slice(&indices).to_device(predictions.device());
    let allowed = action_mask
        .view([-1])
        .index_select(0, &indices)
        .all()
        .int64_value(&[]);
    if allowed == 0 {
        return Err("Q-hat target contains a gate masked by the model".into());
    }
    let targets = Tensor::from_slice(&values)
        .to_device(predictions.device())
        .to_kind(predictions.kind());
    let selected = predictions.view([-1]).index_select(0, &indices);
    Ok(selected.smooth_l1_loss(&targets, Reduction::Mean, 1.0))
}

fn teacher_loss(
    logits: &Tensor,
    action_mask: &Tensor,
    record: &TrainingRecord,
    alphabet: &CnotAlphabet,
    temperature: f64,
) -> Result<(Tensor, bool), String> {
    let mut first_costs: HashMap<CnotGate, u64> = HashMap::new();
    for word in &record.verified_words {
        if let Some(gate) = word.gates.first().copied() {
            first_costs
                .entry(gate)
                .and_modify(|cost| *cost = (*cost).min(word.physical_cost))
                .or_insert(word.physical_cost);
        }
    }
    if first_costs.is_empty() {
        return Ok((
            logits
                .masked_fill(&action_mask.logical_not(), 0.0)
                .sum(logits.kind())
                * 0.0,
            false,
        ));
    }
    let mut gates: Vec<_> = first_costs.into_iter().collect();
    gates.sort_by_key(|(gate, _)| (gate.control, gate.target));
    let best = gates
        .iter()
        .map(|(_, cost)| *cost)
        .min()
        .ok_or_else(|| "teacher gate set unexpectedly empty".to_owned())?;
    let nodes = usize::try_from(logits.size()[1])
        .map_err(|_| "proposal node dimension exceeds host index range")?;
    let mut indices = Vec::with_capacity(gates.len());
    let mut scores = Vec::with_capacity(gates.len());
    for (gate, cost) in &gates {
        if gate.control as usize >= nodes
            || gate.target as usize >= nodes
            || alphabet.cost(*gate).is_none()
        {
            return Err("teacher word starts with an action outside the alphabet".into());
        }
        let index = usize::try_from(gate.control)
            .ok()
            .and_then(|control| control.checked_mul(nodes))
            .and_then(|offset| offset.checked_add(gate.target as usize))
            .ok_or_else(|| "teacher proposal action index overflow".to_owned())?;
        indices.push(i64::try_from(index).map_err(|_| "teacher index exceeds tensor range")?);
        scores.push(-((*cost).saturating_sub(best) as f64) / temperature);
    }
    let indices = Tensor::from_slice(&indices).to_device(logits.device());
    if action_mask
        .view([-1])
        .index_select(0, &indices)
        .all()
        .int64_value(&[])
        == 0
    {
        return Err("teacher action is outside the model's valid action mask".into());
    }
    let target = Tensor::from_slice(&scores)
        .to_device(logits.device())
        .to_kind(logits.kind())
        .softmax(0, logits.kind());
    let selected_log_probs = logits
        .view([-1])
        .index_select(0, &indices)
        .log_softmax(0, logits.kind());
    let loss = -(target * selected_log_probs).sum(logits.kind());
    Ok((loss, true))
}

fn validate_identity(
    identity: &StateIdentity,
    alphabets: &HashMap<String, CnotAlphabet>,
) -> Result<(), String> {
    let alphabet = alphabets
        .get(&identity.alphabet_id)
        .ok_or_else(|| "exposure identity references an unknown alphabet".to_owned())?;
    if identity.dimension != alphabet.dimension() {
        return Err("exposure identity dimension differs from alphabet".into());
    }
    MatrixState::from_packed_rows(
        identity.dimension,
        identity.alphabet_id.clone(),
        identity.packed_rows.clone(),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn sorted_identities(states: &HashSet<StateIdentity>) -> Vec<StateIdentity> {
    let mut result: Vec<_> = states.iter().cloned().collect();
    result.sort_by_key(identity_bytes);
    result
}

fn identity_set_digest(states: &HashSet<StateIdentity>) -> String {
    let mut digest = Sha256::new();
    let mut rows: Vec<_> = states.iter().map(identity_bytes).collect();
    rows.sort();
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        digest.update((row.len() as u64).to_le_bytes());
        digest.update(row);
    }
    hex_digest(digest.finalize().into())
}

fn identity_bytes(identity: &StateIdentity) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&identity.dimension.to_le_bytes());
    bytes.extend_from_slice(&(identity.alphabet_id.len() as u64).to_le_bytes());
    bytes.extend_from_slice(identity.alphabet_id.as_bytes());
    bytes.extend_from_slice(&(identity.packed_rows.len() as u64).to_le_bytes());
    for row in &identity.packed_rows {
        bytes.extend_from_slice(&row.to_le_bytes());
    }
    bytes
}

fn objective_config_digest(
    alphabets: &HashMap<String, CnotAlphabet>,
    catalogs: &HashMap<String, ExactCatalog>,
    policy_id: &str,
    feature_contract_sha256: &str,
    forbidden: &HashSet<StateIdentity>,
    config: &FullGraphObjectiveConfig,
    training_records_sha256: Option<&str>,
) -> String {
    let mut alphabet_rows: Vec<_> = alphabets
        .values()
        .map(|alphabet| {
            serde_json::json!({
                "id": alphabet.id(),
                "dimension": alphabet.dimension(),
                "gates": alphabet.gates().iter().map(|gate| [gate.control, gate.target]).collect::<Vec<_>>(),
                "costs": (0..alphabet.gates().len()).map(|index| alphabet.cost_at(index).unwrap_or_default()).collect::<Vec<_>>()
            })
        })
        .collect();
    alphabet_rows.sort_by_key(|row| row["id"].as_str().unwrap_or_default().to_owned());
    let mut catalog_rows: Vec<_> = catalogs
        .iter()
        .map(|(id, catalog)| (id, catalog.sha256()))
        .collect();
    catalog_rows.sort_by_key(|(id, _)| (*id).clone());
    let body = serde_json::json!({
        "id": "full_graph_reference_v1",
        "policy": policy_id,
        "feature_registry_sha256": feature_contract_sha256,
        "training_records_sha256": training_records_sha256,
        "alphabets": alphabet_rows,
        "exact_sources": catalog_rows,
        "forbidden_sha256": identity_set_digest(forbidden),
        "bellman_weight": config.bellman_weight,
        "supervision_weight": config.supervision_weight,
        "qhat_weight": config.qhat_weight,
        "teacher_weight": config.teacher_weight,
        "neighbor_chunk_size": config.neighbor_chunk_size,
        "teacher_temperature": config.teacher_temperature,
        "external_admission": config.external_admission,
        "external_admission_sha256": config.external_admission_sha256,
        "require_complete_103": config.require_complete_103,
        "compact_exposures": config.compact_exposures,
        "gradient_observation": "neural_trainer_autograd_boundary_v1"
    });
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    hex_digest(Sha256::digest(bytes).into())
}

fn training_records_digest(records: &[TrainingRecord]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/ordered_training_records_v1\0");
    digest.update((records.len() as u64).to_le_bytes());
    for record in records {
        let state_identity = StateIdentity::from(&record.state);
        hash_digest_bytes(&mut digest, &identity_bytes(&state_identity));
        digest.update([match record.split {
            crate::training_data::TrainingSplit::Train => 0,
            crate::training_data::TrainingSplit::Dev => 1,
            crate::training_data::TrainingSplit::Test => 2,
        }]);
        hash_digest_bytes(&mut digest, record.lineage_id.as_bytes());
        match record.exact_distance {
            Some(distance) => {
                digest.update([1]);
                digest.update(distance.to_le_bytes());
            }
            None => digest.update([0]),
        }
        digest.update((record.verified_words.len() as u64).to_le_bytes());
        for certificate in &record.verified_words {
            let start = identity_bytes(&StateIdentity::from(&certificate.start));
            let end = identity_bytes(&StateIdentity::from(&certificate.end));
            hash_digest_bytes(&mut digest, &start);
            hash_digest_bytes(&mut digest, &end);
            digest.update((certificate.gates.len() as u64).to_le_bytes());
            for gate in &certificate.gates {
                digest.update(gate.control.to_le_bytes());
                digest.update(gate.target.to_le_bytes());
            }
            digest.update(certificate.physical_cost.to_le_bytes());
            hash_digest_bytes(&mut digest, certificate.producer.as_bytes());
            hash_digest_bytes(&mut digest, certificate.lineage_id.as_bytes());
        }
    }
    hex_digest(digest.finalize().into())
}

fn hash_digest_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hex_digest(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
