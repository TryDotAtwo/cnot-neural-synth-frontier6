use super::{ExposureAudit, FullGraphObjective, ObjectiveDiagnostics, EXPOSURE_SCHEMA};

impl FullGraphObjective {
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
}
