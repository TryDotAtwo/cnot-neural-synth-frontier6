use super::{ExposureAudit, FullGraphObjective, ObjectiveDiagnostics, EXPOSURE_SCHEMA};
use crate::nn_training::ADAMW_UPDATE_SEMANTICS_ID;

impl FullGraphObjective {
    #[cfg(feature = "cuda")]
    pub(crate) fn alphabet_for(&self, id: &str) -> Result<&crate::CnotAlphabet, String> {
        self.alphabets
            .get(id)
            .ok_or_else(|| format!("unknown objective alphabet {id}"))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn feature_policy(&self) -> &crate::FeaturePolicy {
        &self.policy
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn diagnostics(&self) -> &ObjectiveDiagnostics {
        &self.last_diagnostics
    }

    pub fn audit(&self) -> ExposureAudit {
        let mut audit = self.exposures.audit(&self.forbidden, self.config.compact_exposures);
        if self.config.prevalidated_device_closure {
            audit.strict_unseen &= self.closure_verified;
            audit.audit_mode = if self.closure_verified {
                "prevalidated_device_one_step_closure"
            } else {
                "prevalidated_device_closure_pending"
            };
            if self.closure_verified {
                audit.set_sha256.insert("closure_contract".into(), self.config_sha256.clone());
                audit.set_sha256.insert("forbidden".into(), super::fingerprint::identity_set_digest(&self.forbidden));
            }
        }
        audit
    }

    /// Stable checkpoint contract for `Checkpoint.contract`.
    pub fn checkpoint_contract(&self) -> serde_json::Value {
        serde_json::json!({
            "id": "full_graph_reference_v3",
            "objective_sha256": self.config_sha256,
            "adamw_update_semantics": ADAMW_UPDATE_SEMANTICS_ID,
            "feature_registry_sha256": self.feature_contract_sha256,
            "frozen_dataset_sha256": self.frozen_dataset_sha256,
            "training_records_sha256": self.training_records_sha256,
            "exposure_schema": EXPOSURE_SCHEMA,
        })
    }
}
