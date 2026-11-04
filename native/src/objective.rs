//! Source-aligned full-graph objective and exposure accounting.
//!
//! Training records are validated as one batch before labels, feature forwards,
//! or gradients are admitted. Exact labels require a verified closed catalog;
//! verified reduction words are upper bounds only.

use crate::alphabet::CnotAlphabet;
use crate::features::FeaturePolicy;
use crate::packet::StateIdentity;
use crate::training_data::ExactCatalog;
use std::collections::{HashMap, HashSet};

mod binding;
mod construction;
mod contract;
mod exposure;
mod fingerprint;
mod losses;
mod runner;
mod state;
mod training;

pub use exposure::{ExposureAudit, ObjectiveDiagnostics, StateAdmission};
use exposure::{ExposureLedger, EXPOSURE_SCHEMA};
use fingerprint::{
    is_sha256_hex, objective_config_digest, training_records_digest, validate_identity,
    ObjectiveFingerprintInput,
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
