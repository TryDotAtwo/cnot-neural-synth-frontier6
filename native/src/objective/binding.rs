use super::{is_sha256_hex, objective_config_digest, training_records_digest, FullGraphObjective};
use crate::training_data::TrainingRecord;

impl FullGraphObjective {
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

    pub(super) fn bind_training_records(
        &mut self,
        records: &[TrainingRecord],
    ) -> Result<(), String> {
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
