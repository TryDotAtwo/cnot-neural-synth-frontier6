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
#[cfg(feature = "cuda")]
pub use runner::device::{DeviceDataParallelRunner, DeviceFullGraphRunner};
pub use runner::FullGraphObjectiveRunner;

#[derive(Clone, Debug)]
pub struct FullGraphObjectiveConfig {
    pub bellman_weight: f64,
    pub supervision_weight: f64,
    pub qhat_weight: f64,
    pub teacher_weight: f64,
    pub neighbor_chunk_size: usize,
    /// Experimental approximate Bellman objective. Half the actions are chosen
    /// by Q-hat and half by a deterministic rotating exploration schedule.
    /// None keeps the original complete primitive backup.
    pub sampled_neighbor_count: Option<usize>,
    /// Bound nonterminal candidate chunks by feature workspace; terminal Q-hat
    /// retains the existing single-packet semantics.
    pub adaptive_neighbor_chunk: bool,
    /// Factor the wire terms in the detached candidate scoring pass.
    pub share_candidate_wire_projections: bool,
    /// Factor the same affine map in gradient-carrying forwards at or above
    /// this width. Floating-point summation order changes, so this is opt-in.
    pub share_training_wire_projections_min_width: Option<u32>,
    /// Cache immutable root F08 results on the GPU; successor inputs are still
    /// computed for each Bellman backup.
    pub cache_root_f08: bool,
    /// Compute a logical batch one graph at a time and accumulate gradients.
    /// This preserves one optimizer update but changes FP32 summation order.
    pub sequential_batch_gradients: bool,
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
    /// The device runner proves that every frozen root and its entire primitive
    /// one-step closure avoid the forbidden set before any model input.
    pub prevalidated_device_closure: bool,
}

impl Default for FullGraphObjectiveConfig {
    fn default() -> Self {
        Self {
            bellman_weight: 1.0,
            supervision_weight: 1.0,
            qhat_weight: 0.1,
            teacher_weight: 0.0,
            neighbor_chunk_size: 32,
            sampled_neighbor_count: None,
            adaptive_neighbor_chunk: false,
            share_candidate_wire_projections: false,
            share_training_wire_projections_min_width: None,
            cache_root_f08: false,
            sequential_batch_gradients: false,
            teacher_temperature: 4.0,
            external_admission: false,
            external_admission_sha256: None,
            require_all_model_inputs: true,
            compact_exposures: false,
            prevalidated_device_closure: false,
        }
    }
}

impl FullGraphObjectiveConfig {
    fn validate_policy(&self, policy: &FeaturePolicy) -> Result<(), String> {
        if self.cache_root_f08
            && (policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
                || policy.evaluations.is_some() || !self.require_all_model_inputs)
        {
            return Err("root F08 cache requires the complete missing-token schema and unlimited evaluations".into());
        }
        if self.prevalidated_device_closure
            && policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
        {
            return Err("prevalidated device closure requires the missing-token schema".into());
        }
        if self.share_training_wire_projections_min_width.is_some()
            && policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
        {
            return Err("shared training projections require the missing-token schema".into());
        }
        if self.sampled_neighbor_count.is_some()
            && (policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
                || policy.evaluations.is_some()
                || !self.require_all_model_inputs)
        {
            return Err("sampled neighbors require the missing-token schema, unlimited feature evaluations, and all model inputs".into());
        }
        if self.adaptive_neighbor_chunk
            && (policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
                || policy.evaluations.is_some()
                || !self.require_all_model_inputs)
        {
            return Err("adaptive neighbor chunks require the missing-token schema, unlimited feature evaluations, and all model inputs".into());
        }
        Ok(())
    }

    fn effective_neighbor_chunk_size(
        &self,
        policy: &FeaturePolicy,
        n: usize,
    ) -> Result<usize, String> {
        self.validate_policy(policy)?;
        if !self.adaptive_neighbor_chunk {
            return Ok(self.neighbor_chunk_size);
        }
        let Some(limit) = policy.max_bytes else {
            return Ok(self.neighbor_chunk_size);
        };
        let per_row = crate::features::workspace_bytes_per_state(n)
            .filter(|&bytes| bytes > 0)
            .ok_or("feature workspace estimate overflow or zero")?;
        let capacity = limit / per_row;
        if capacity == 0 {
            return Err(format!(
                "feature workspace limit cannot fit one width-{n} state"
            ));
        }
        Ok(self.neighbor_chunk_size.min(capacity))
    }

    fn validate(&self) -> Result<(), String> {
        if self.prevalidated_device_closure
            && (!self.compact_exposures || self.external_admission)
        {
            return Err("prevalidated device closure requires compact exposures and no external admission".into());
        }
        if self.share_training_wire_projections_min_width == Some(0) {
            return Err("shared training projection minimum width must be positive".into());
        }
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
        if self.sampled_neighbor_count.is_some_and(|count| count < 2 || count > i64::MAX as usize)
            || self.sampled_neighbor_count.is_some() && self.qhat_weight <= 0.0
        {
            return Err("sampled neighbors require at least two candidates and a positive Q-hat loss weight".into());
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
    /// Runtime proof only; never restored from checkpoint data.
    closure_verified: bool,
    last_diagnostics: ObjectiveDiagnostics,
}

#[cfg(test)]
mod adaptive_chunk_tests {
    use super::*;

    #[test]
    fn adaptive_chunk_respects_feature_workspace_and_policy_boundaries() {
        let mut policy = FeaturePolicy::all_model_inputs(crate::schema::MISSING_TOKEN_LAYOUT_ID);
        policy.max_bytes = Some(256 * 1024 * 1024);
        let mut config = FullGraphObjectiveConfig::default();

        // Existing manifests retain the exact fixed-size behavior.
        assert_eq!(
            config.effective_neighbor_chunk_size(&policy, 144).unwrap(),
            32
        );
        config.adaptive_neighbor_chunk = true;
        assert_eq!(
            config.effective_neighbor_chunk_size(&policy, 72).unwrap(),
            32
        );
        assert_eq!(
            config.effective_neighbor_chunk_size(&policy, 144).unwrap(),
            12
        );
        policy.max_bytes = None;
        assert_eq!(
            config.effective_neighbor_chunk_size(&policy, 144).unwrap(),
            32
        );
        policy.max_bytes = Some(1);
        assert!(config.effective_neighbor_chunk_size(&policy, 144).is_err());
        policy.max_bytes = Some(256 * 1024 * 1024);
        assert!(config
            .effective_neighbor_chunk_size(&policy, usize::MAX)
            .is_err());
        policy.evaluations = Some(100);
        assert!(config.effective_neighbor_chunk_size(&policy, 72).is_err());
        policy.evaluations = None;
        policy.schema_id = "runtime_packet_layout_v24".into();
        assert!(config.effective_neighbor_chunk_size(&policy, 72).is_err());
        policy.schema_id = crate::schema::MISSING_TOKEN_LAYOUT_ID.into();
        config.require_all_model_inputs = false;
        assert!(config.effective_neighbor_chunk_size(&policy, 72).is_err());
    }
}
