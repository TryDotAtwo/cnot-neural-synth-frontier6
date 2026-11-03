use super::{FeatureEntry, FeaturePacket, FeaturePacketError};
use crate::schema::FeatureId;

impl FeaturePacket {
    /// Copy one immutable batch row while preserving its feature masks and provenance.
    pub fn single_row(&self, row: usize) -> Result<Self, FeaturePacketError> {
        if self.batch == 0 || row >= self.batch {
            return Err(FeaturePacketError(
                "feature row is outside the packet".into(),
            ));
        }
        if self.state_keys.len() != self.batch
            || self.alphabet_ids.len() != self.batch
            || self.valid_nodes.len() != self.batch.checked_mul(self.nodes).unwrap_or(usize::MAX)
        {
            return Err(FeaturePacketError(
                "feature packet batch metadata is malformed".into(),
            ));
        }

        let mut entries = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if entry.shape.first().copied() != Some(self.batch)
                || entry.available.len() != self.batch
                || entry.status.len() != self.batch
                || entry.reason.len() != self.batch
                || entry.provenance.len() != self.batch
                || entry.values.len() % self.batch != 0
            {
                return Err(FeaturePacketError(format!(
                    "{} cannot be sliced from a malformed batch entry",
                    entry.spec.id
                )));
            }
            let values_per_row = entry.values.len() / self.batch;
            let values_start = row
                .checked_mul(values_per_row)
                .ok_or_else(|| FeaturePacketError("feature row offset overflow".into()))?;
            let values_end = values_start
                .checked_add(values_per_row)
                .ok_or_else(|| FeaturePacketError("feature row end overflow".into()))?;
            let mut shape = entry.shape.clone();
            shape[0] = 1;
            let sliced = FeatureEntry {
                spec: entry.spec.clone(),
                shape,
                values: entry
                    .values
                    .get(values_start..values_end)
                    .ok_or_else(|| FeaturePacketError("feature row is truncated".into()))?
                    .to_vec(),
                available: vec![entry.available[row]],
                status: vec![entry.status[row]],
                reason: vec![entry.reason[row].clone()],
                provenance: vec![entry.provenance[row].clone()],
                wire_roles: entry.wire_roles.clone(),
                token_valid: slice_optional_row(&entry.token_valid, self.batch, row)?,
                incidence: slice_optional_row(&entry.incidence, self.batch, row)?,
            };
            sliced.validate(1, self.nodes)?;
            entries.push(sliced);
        }

        let nodes_start = row
            .checked_mul(self.nodes)
            .ok_or_else(|| FeaturePacketError("node-mask row offset overflow".into()))?;
        let nodes_end = nodes_start
            .checked_add(self.nodes)
            .ok_or_else(|| FeaturePacketError("node-mask row end overflow".into()))?;
        Ok(Self {
            entries,
            state_keys: vec![self.state_keys[row].clone()],
            alphabet_ids: vec![self.alphabet_ids[row].clone()],
            batch: 1,
            nodes: self.nodes,
            valid_nodes: self
                .valid_nodes
                .get(nodes_start..nodes_end)
                .ok_or_else(|| FeaturePacketError("node-mask row is truncated".into()))?
                .to_vec(),
            schema_id: self.schema_id.clone(),
            policy_id: self.policy_id.clone(),
            feature_id_manifest_id: self.feature_id_manifest_id,
        })
    }

    pub fn feature(&self, id: FeatureId) -> &FeatureEntry {
        &self.entries[id.index()]
    }
}

fn slice_optional_row<T: Clone>(
    values: &Option<Vec<T>>,
    batch: usize,
    row: usize,
) -> Result<Option<Vec<T>>, FeaturePacketError> {
    match values {
        None => Ok(None),
        Some(values) if values.len() % batch == 0 => {
            let stride = values.len() / batch;
            let start = row
                .checked_mul(stride)
                .ok_or_else(|| FeaturePacketError("group row offset overflow".into()))?;
            let end = start
                .checked_add(stride)
                .ok_or_else(|| FeaturePacketError("group row end overflow".into()))?;
            values
                .get(start..end)
                .map(|slice| Some(slice.to_vec()))
                .ok_or_else(|| FeaturePacketError("group metadata is truncated".into()))
        }
        Some(_) => Err(FeaturePacketError(
            "group metadata batch length differs".into(),
        )),
    }
}
