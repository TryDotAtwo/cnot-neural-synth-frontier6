//! The reference spends a provider evaluation in canonical feature order for
//! each physical row, including an unsupported or inapplicable result. Builtins
//! are free. At most one feature can therefore cover a partial row prefix.
use super::*;
use crate::FeatureRole;

impl DeviceFeaturePlan {
    // Trained search uses the same frozen evaluation policy as the Bellman objective.
    // A limited final packet spends workspace on actual candidates; the beam also
    // carries fixed padding whose extent is not part of the source policy.
    pub(crate) fn search_packet_capacity(&self, n: u32) -> Result<i64, String> {
        if self.specs.len() != crate::schema::MODEL_INPUT_COUNT || self.evaluations.is_some() {
            return Err("learned search requires the complete model-input plan and frozen unlimited deterministic evaluations".into());
        }
        let Some(bytes) = self.max_bytes else {
            return Ok(i64::MAX);
        };
        let per_row = crate::features::workspace_bytes_per_state(n as usize)
            .ok_or("feature workspace estimate overflow")?;
        Ok(i64::try_from(bytes / per_row).unwrap_or(i64::MAX))
    }

    pub(crate) fn fits_workspace(&self, n: u32, rows: usize) -> Result<bool, String> {
        let bytes = crate::features::workspace_bytes_per_state(n as usize)
            .and_then(|v| v.checked_mul(rows))
            .ok_or("feature workspace estimate overflow")?;
        Ok(self.max_bytes.is_none_or(|limit| bytes <= limit))
    }

    pub(super) fn scheduled(
        &self,
        states: &DeviceStateBatch,
        cached_f08: Option<&DeviceFeatureEntry>,
    ) -> Result<(Tensor, Vec<DeviceFeatureEntry>), String> {
        let rows = usize::try_from(states.batch_size()).map_err(|_| "invalid GPU packet batch")?;
        let fits = self.fits_workspace(states.dimension(), rows)?;
        if fits
            && self
                .evaluations
                .is_none_or(|limit| limit / rows >= self.provider_count)
        {
            return self.compute(states, &self.specs, cached_f08);
        }
        let n = i64::from(states.dimension());
        let device = states.tensor().device();
        let mut active = Vec::new();
        let mut partial = None;
        let mut entries = Vec::with_capacity(self.specs.len());
        let mut remaining = self.evaluations.unwrap_or(usize::MAX);
        for spec in &self.specs {
            let allowed = if !fits {
                0
            } else if crate::structural::is_builtin(spec.id) {
                rows
            } else {
                let count = remaining.min(rows);
                remaining -= count;
                count
            };
            if allowed == rows {
                active.push(spec.clone());
            } else if allowed == 0 {
                entries.push(unavailable(
                    spec,
                    states.batch_size(),
                    n,
                    device,
                    DeviceFeatureStatus::ResourceLimit,
                ));
            } else {
                partial = Some((spec, allowed as i64));
            }
        }
        let matrix = if active.is_empty() {
            crate::device_state::unpack_matrix(states.tensor(), n).to_kind(Kind::Float)
        } else {
            let (matrix, computed) = self.compute(states, &active, cached_f08)?;
            entries.extend(computed);
            matrix
        };
        if let Some((spec, count)) = partial {
            let (_, mut computed) =
                self.compute(&states.rows(0, count)?, std::slice::from_ref(spec), None)?;
            if computed.len() != 1 || computed[0].spec() != spec {
                return Err(format!(
                    "{} partial provider returned a different contract",
                    spec.id
                ));
            }
            entries.push(pad_prefix(computed.remove(0), states.batch_size() - count));
        }
        Ok((matrix, entries))
    }
}

pub(super) fn unavailable(
    spec: &FeatureSpec,
    rows: i64,
    n: i64,
    device: Device,
    status: DeviceFeatureStatus,
) -> DeviceFeatureEntry {
    let c = spec.channels as i64;
    let shape = match spec.role {
        FeatureRole::Node => vec![rows, n, c],
        FeatureRole::Pair => vec![rows, n, n, c],
        FeatureRole::Global => vec![rows, c],
        FeatureRole::Group => vec![rows, 0, c],
    };
    let group = spec.role == FeatureRole::Group;
    DeviceFeatureEntry::new(
        spec.clone(),
        Tensor::zeros(&shape, (Kind::Float, device)),
        Tensor::full([rows], status as i64, (Kind::Uint8, device)),
        group.then(|| Tensor::zeros([rows, 0], (Kind::Bool, device))),
        group.then(|| {
            Tensor::zeros(
                [rows, 0, n, spec.wire_roles.len() as i64],
                (Kind::Float, device),
            )
        }),
    )
}

fn pad_prefix(entry: DeviceFeatureEntry, denied: i64) -> DeviceFeatureEntry {
    let extend = |t: &Tensor, fill: i64| {
        let mut shape = t.size();
        shape[0] = denied;
        Tensor::cat(&[t, &Tensor::full(shape, fill, (t.kind(), t.device()))], 0)
    };
    DeviceFeatureEntry::new(
        entry.spec().clone(),
        extend(entry.values(), 0),
        extend(entry.status(), DeviceFeatureStatus::ResourceLimit as i64),
        entry.token_valid().map(|m| extend(m, 0)),
        entry.incidence().map(|v| extend(v, 0)),
    )
}
