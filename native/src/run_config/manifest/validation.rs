use super::RunManifest;

impl RunManifest {
    pub(super) fn validate(&self) -> Result<(), String> {
        validate_sha256(&self.dataset.sha256, "dataset")?;
        if let Some(catalogs) = &self.exact_catalogs {
            validate_sha256(&catalogs.sha256, "exact catalogs")?;
        }
        if self.checkpoint_dir.as_os_str().is_empty() {
            return Err("checkpoint_dir must not be empty".into());
        }
        if self.training.attempts == 0
            || self.training.target_attempts == Some(0)
            || self.training.batch_size == 0
            || self.training.checkpoint_every_attempts == 0
        {
            return Err(
                "attempts, batch_size, and checkpoint_every_attempts must be positive; target_attempts, when set, must be positive".into(),
            );
        }
        validate_cuda_devices(
            self.training.cuda_device,
            &self.training.cuda_devices,
            self.training.batch_size,
        )?;
        if self.feature_policy.schema_id.trim().is_empty() {
            return Err("feature_policy.schema_id must be nonempty".into());
        }
        if self.objective.adaptive_neighbor_chunk
            && (self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
                || self.feature_policy.evaluations.0.is_some())
        {
            return Err("adaptive neighbor chunks require the missing-token schema and unlimited feature evaluations".into());
        }
        if self.objective.cache_root_f08
            && (self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
                || self.feature_policy.evaluations.0.is_some())
        {
            return Err("root F08 cache requires the missing-token schema and unlimited feature evaluations".into());
        }
        if self.objective.sampled_neighbor_count.is_some()
            && (self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID
                || self.feature_policy.evaluations.0.is_some())
        {
            return Err("sampled neighbors require the missing-token schema and unlimited feature evaluations".into());
        }
        if self.objective.sampled_neighbor_count.is_some_and(|n| n < 2 || n > i64::MAX as usize)
            || self.objective.sampled_neighbor_count.is_some() && self.objective.qhat_weight <= 0.0
        {
            return Err("sampled neighbors require at least two candidates and a positive Q-hat loss weight".into());
        }
        if self.alphabets.is_empty() {
            return Err("at least one exact alphabet must be provided".into());
        }
        if self.feature_contexts.max_n < 2 {
            return Err("feature_contexts.max_n must be at least two".into());
        }
        if let Some(max_n) = self.feature_contexts.f08_max_n {
            if max_n < 2 || max_n > self.feature_contexts.max_n {
                return Err("feature_contexts.f08_max_n must be between two and max_n".into());
            }
            if self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID {
                return Err("feature_contexts.f08_max_n requires the missing-token schema".into());
            }
        }
        if let Some(limit) = self.feature_contexts.f08_max_changed_rows {
            if limit == 0 || limit > self.feature_contexts.f08_max_n() {
                return Err("feature_contexts.f08_max_changed_rows must be between one and f08_max_n".into());
            }
            if self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID {
                return Err("feature_contexts.f08_max_changed_rows requires the missing-token schema".into());
            }
        }
        if let Some(limit) = self.feature_contexts.f08_max_component_events {
            if limit == 0 || u64::from(limit) > u64::from(self.feature_contexts.f08_max_n()) + 1 {
                return Err("feature_contexts.f08_max_component_events must be between one and f08_max_n plus one".into());
            }
            if self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID {
                return Err("feature_contexts.f08_max_component_events requires the missing-token schema".into());
            }
        }
        if let Some(limit) = self.feature_contexts.f08_max_history_counts {
            if limit == 0 {
                return Err("feature_contexts.f08_max_history_counts must be positive".into());
            }
            if self.feature_policy.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID {
                return Err("feature_contexts.f08_max_history_counts requires the missing-token schema".into());
            }
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

fn validate_cuda_devices(
    master: usize,
    devices: &[usize],
    batch_size: usize,
) -> Result<(), String> {
    if devices.is_empty() {
        return Ok(());
    }
    if !matches!(devices.len(), 1 | 2)
        || devices[0] != master
        || devices
            .iter()
            .enumerate()
            .any(|(index, device)| devices[..index].contains(device))
    {
        return Err(
            "cuda_devices must name one or two distinct devices, with cuda_device as the master"
                .into(),
        );
    }
    if devices.len() == 2 && batch_size < 2 {
        return Err("two CUDA devices require batch_size of at least two".into());
    }
    Ok(())
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

#[cfg(test)]
mod cuda_device_tests {
    use super::validate_cuda_devices;

    #[test]
    fn accepts_single_dual_and_legacy_device_selection() {
        assert!(validate_cuda_devices(0, &[0], 4).is_ok());
        assert!(validate_cuda_devices(0, &[0, 1], 4).is_ok());
        assert!(validate_cuda_devices(0, &[], 4).is_ok());
    }

    #[test]
    fn rejects_mismatched_duplicate_oversized_and_underbatched_selections() {
        assert!(validate_cuda_devices(0, &[1], 4).is_err());
        assert!(validate_cuda_devices(0, &[0, 0], 4).is_err());
        assert!(validate_cuda_devices(0, &[0, 1, 2], 4).is_err());
        assert!(validate_cuda_devices(0, &[0, 1], 1).is_err());
    }
}
