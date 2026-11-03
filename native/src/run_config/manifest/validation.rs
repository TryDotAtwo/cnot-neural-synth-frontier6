use super::RunManifest;

impl RunManifest {
    pub(super) fn validate(&self) -> Result<(), String> {
        validate_sha256(&self.dataset.sha256, "dataset")?;
        validate_sha256(&self.exact_catalogs.sha256, "exact catalogs")?;
        if self.checkpoint_dir.as_os_str().is_empty() {
            return Err("checkpoint_dir must not be empty".into());
        }
        if self.training.attempts == 0
            || self.training.batch_size == 0
            || self.training.checkpoint_every_attempts == 0
        {
            return Err(
                "attempts, batch_size, and checkpoint_every_attempts must be positive".into(),
            );
        }
        if self.feature_policy.schema_id.trim().is_empty() {
            return Err("feature_policy.schema_id must be nonempty".into());
        }
        if self.alphabets.is_empty() {
            return Err("at least one exact alphabet must be provided".into());
        }
        if self.feature_contexts.max_n < 2 {
            return Err("feature_contexts.max_n must be at least two".into());
        }
        let w = [
            self.objective.bellman_weight,
            self.objective.supervision_weight,
            self.objective.qhat_weight,
            self.objective.teacher_weight,
        ];
        if w.iter().any(|x| !x.is_finite() || *x < 0.0)
            || w.iter().sum::<f64>() <= 0.0
            || self.objective.neighbor_chunk_size == 0
            || !self.objective.teacher_temperature.is_finite()
            || self.objective.teacher_temperature <= 0.0
        {
            return Err("objective weights and budgets are invalid".into());
        }
        if self.feature_policy.evaluations.0 == Some(0)
            || self.feature_policy.max_bytes.0 == Some(0)
        {
            return Err("feature evaluation and byte limits, when set, must be positive".into());
        }
        Ok(())
    }
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!(
            "{label} SHA-256 must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}
