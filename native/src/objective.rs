//! Source-aligned full-graph objective and exposure accounting.
//!
//! Training records are validated as one batch before labels, feature forwards,
//! or gradients are admitted. Exact labels require a verified closed catalog;
//! verified reduction words are upper bounds only.

use crate::alphabet::CnotAlphabet;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::packet::StateIdentity;
use crate::schema::all_model_input_specs;
use crate::state::MatrixState;
use crate::training_data::{ExactCatalog, TrainingRecord};
use std::collections::{HashMap, HashSet};

mod exposure;
mod fingerprint;
mod losses;
mod runner;
mod state;
mod training;

use exposure::{EXPOSURE_SCHEMA, ExposureLedger};
pub use exposure::{ExposureAudit, ObjectiveDiagnostics, StateAdmission};
use fingerprint::{
    is_sha256_hex, objective_config_digest, training_records_digest, validate_identity,
};
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
                return Err("admission fingerprint supplied without external admission".into());
            }
            (true, None) => {
                return Err("external admission requires a stable SHA-256 fingerprint".into());
            }
            (true, Some(_)) => {
                return Err("external admission fingerprint must be lowercase SHA-256 hex".into());
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
}
