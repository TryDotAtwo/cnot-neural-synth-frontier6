use super::{DeviceFeatureEntry, DeviceFeaturePacket, DeviceFeatureStatus, ModelError};
use crate::model::tensor_packet::TensorPacket;
use crate::schema::MODEL_INPUT_COUNT;
use tch::Tensor;

impl DeviceFeaturePacket {
    /// In the scalar providers every physical group contributes one valid token.
    /// A source packet's group axis is nonempty if any row produced such a token.
    /// single_row preserves that chunk-wide axis, even for an unavailable winner.
    pub(crate) fn group_activity(&self) -> Tensor {
        let present: Vec<_> = self
            .tensors
            .entries
            .iter()
            .flatten()
            .filter_map(|entry| {
                entry.token_valid.as_ref().map(|mask| {
                    mask.to_kind(tch::Kind::Bool)
                        .logical_and(
                            &entry
                                .status
                                .eq(DeviceFeatureStatus::Computed as i64)
                                .unsqueeze(1),
                        )
                        .any()
                })
            })
            .collect();
        if present.is_empty() {
            Tensor::empty([0], (tch::Kind::Bool, self.matrix().device()))
        } else {
            Tensor::stack(&present, 0)
        }
    }

    /// The constructor has already enforced unique, canonical slots and excluded H07.
    pub(crate) fn complete_coverage(&self) -> Result<Tensor, ModelError> {
        let statuses = self.selected_status_matrix()?;
        Ok(self.admissible_statuses(&statuses).all())
    }

    fn admissible_statuses(&self, statuses: &Tensor) -> Tensor {
        if self.schema_id != crate::schema::MISSING_TOKEN_LAYOUT_ID {
            return statuses
                .eq(DeviceFeatureStatus::Computed as i64)
                .logical_or(&statuses.eq(DeviceFeatureStatus::Inapplicable as i64));
        }
        let optional = statuses.narrow(0, 1, MODEL_INPUT_COUNT as i64 - 1);
        let valid = optional
            .ge(DeviceFeatureStatus::Computed as i64)
            .logical_and(&optional.le(DeviceFeatureStatus::ResourceLimit as i64));
        let matrix = statuses
            .narrow(0, 0, 1)
            .eq(DeviceFeatureStatus::Computed as i64);
        Tensor::cat(&[matrix, valid], 0)
    }

    /// Zero admits a row. Otherwise the high bits identify the first failing
    /// canonical input slot (one-based) and the low byte its exact status.
    pub(crate) fn first_coverage_failure(&self) -> Result<Tensor, ModelError> {
        let statuses = self.selected_status_matrix()?.to_kind(tch::Kind::Int64);
        let bad = self.admissible_statuses(&statuses).logical_not();
        let slot = bad.to_kind(tch::Kind::Int64).argmax(0, false);
        let code = (&slot + 1) * 256 + statuses.gather(0, &slot.unsqueeze(0), false).squeeze_dim(0);
        Ok(code * bad.any_dim(0, false))
    }

    pub(crate) fn selected_status_matrix(&self) -> Result<Tensor, ModelError> {
        let statuses: Vec<_> = self
            .tensors
            .entries
            .iter()
            .flatten()
            .map(|e| &e.status)
            .collect();
        if statuses.len() != MODEL_INPUT_COUNT {
            return Err(ModelError(format!(
                "complete coverage requires all {MODEL_INPUT_COUNT} model inputs"
            )));
        }
        Ok(Tensor::stack(&statuses, 0))
    }

    /// Keep just one winning row. The decision and every selected value, status,
    /// token mask and physical incidence stay on the device. No chunk is held
    /// alive by a view of its winning row.
    /// Freeze the selected row before deriving optimizer participation. Rows that
    /// lose this chunk's detached minimum must not activate their AdamW groups.
    pub(crate) fn freeze_winner_with_activity(
        &self,
        local_index: &Tensor,
        incumbent: Option<(&Self, &Tensor)>,
    ) -> (Self, Tensor) {
        let packet = self.freeze_winner(local_index, incumbent);
        let activity = packet.group_activity();
        (packet, activity)
    }

    pub(crate) fn freeze_winner(
        &self,
        local_index: &Tensor,
        incumbent: Option<(&Self, &Tensor)>,
    ) -> Self {
        let index = local_index.view([1]);
        let pick = |candidate: &Tensor, old: Option<&Tensor>| {
            let row = candidate.index_select(0, &index);
            match (incumbent, old) {
                (Some((_, replace)), Some(old)) if row.size() != old.size() => {
                    // A resource-limited chunk may have fewer padded group tokens.
                    // All other axes are fixed by the common producer and physical width.
                    let width = row.size()[1].max(old.size()[1]);
                    pad_groups(&row, width).where_self(replace, &pad_groups(old, width))
                }
                (Some((_, replace)), Some(old)) => row.where_self(replace, old),
                _ => row,
            }
        };
        let previous = incumbent.map(|(packet, _)| &packet.tensors);
        let entries = self
            .tensors
            .entries
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                let entry = entry.as_ref()?;
                let old = previous
                    .map(|packet| packet.entries[slot].as_ref().expect("same feature plan"));
                Some(DeviceFeatureEntry {
                    spec: entry.spec.clone(),
                    values: pick(&entry.values, old.map(|e| &e.values)),
                    status: pick(&entry.status, old.map(|e| &e.status)),
                    token_valid: entry
                        .token_valid
                        .as_ref()
                        .map(|t| pick(t, old.and_then(|e| e.token_valid.as_ref()))),
                    incidence: entry
                        .incidence
                        .as_ref()
                        .map(|t| pick(t, old.and_then(|e| e.incidence.as_ref()))),
                })
            })
            .collect();
        Self {
            tensors: TensorPacket {
                matrix: pick(self.matrix(), previous.map(|p| &p.matrix)),
                valid_nodes: pick(self.valid_nodes(), previous.map(|p| &p.valid_nodes)),
                entries,
            },
            schema_id: self.schema_id.clone(),
            policy_id: self.policy_id.clone(),
        }
    }
}

fn pad_groups(rows: &Tensor, capacity: i64) -> Tensor {
    let mut shape = rows.size();
    shape[1] = capacity - shape[1];
    if shape[1] == 0 {
        rows.shallow_clone()
    } else {
        Tensor::cat(
            &[rows, &Tensor::zeros(shape, (rows.kind(), rows.device()))],
            1,
        )
    }
}

#[cfg(test)]
mod missing_status_tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn missing_layout_admits_optional_reasons_but_requires_a01() {
        crate::torch_device::require_cuda_device(0).unwrap();
        let device = Device::Cuda(0);
        let packet = DeviceFeaturePacket {
            tensors: TensorPacket {
                matrix: Tensor::zeros([1, 1, 1], (Kind::Float, device)),
                valid_nodes: Tensor::ones([1, 1], (Kind::Bool, device)),
                entries: Vec::new(),
            },
            schema_id: crate::schema::MISSING_TOKEN_LAYOUT_ID.to_owned(),
            policy_id: "missing-status-test".to_owned(),
        };
        let batch = 1 + MODEL_INPUT_COUNT * 3 + 2;
        let mut statuses = vec![DeviceFeatureStatus::Computed as u8; MODEL_INPUT_COUNT * batch];
        for feature in 0..MODEL_INPUT_COUNT {
            for reason in 0..3 {
                statuses[feature * batch + 1 + feature * 3 + reason] =
                    DeviceFeatureStatus::Inapplicable as u8 + reason as u8;
            }
        }
        statuses[batch + batch - 2] = DeviceFeatureStatus::NotRequested as u8;
        statuses[batch + batch - 1] = 5;
        let statuses = Tensor::from_slice(&statuses)
            .view([MODEL_INPUT_COUNT as i64, batch as i64])
            .to_device(device);
        let admitted = packet.admissible_statuses(&statuses).all_dim(0, false)
            .to_device(Device::Cpu).to_kind(Kind::Int64);
        let admitted = Vec::<i64>::try_from(admitted).unwrap();
        assert_eq!(admitted[0], 1);
        assert_eq!(&admitted[1..4], &[0, 0, 0]);
        assert!(admitted[4..batch-2].iter().all(|value| *value == 1));
        assert_eq!(&admitted[batch-2..], &[0, 0]);
    }
}


#[cfg(test)]
mod winner_activity_tests {
    use super::*;
    use crate::schema::{FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT};
    use tch::{Device, Kind};

    fn packet(token_valid: &[u8]) -> DeviceFeaturePacket {
        let device = Device::Cpu;
        let batch = token_valid.len() as i64;
        let id = FeatureId::new(b'B', 1).unwrap();
        let spec = FeatureSpec {
            id,
            role: FeatureRole::Group,
            channels: 1,
            scope: "winner-activity-test".into(),
            producer_version: "test".into(),
            dependencies: Vec::new(),
            exactness: "exact".into(),
            cost_tier: "T1".into(),
            wire_roles: vec!["source".into()],
            requires_solver_assisted: false,
        };
        let mut entries: Vec<_> = (0..FEATURE_COUNT).map(|_| None).collect();
        entries[id.index()] = Some(DeviceFeatureEntry::new(
            spec,
            Tensor::zeros([batch, 1, 1], (Kind::Float, device)),
            Tensor::full(
                [batch],
                DeviceFeatureStatus::Computed as i64,
                (Kind::Uint8, device),
            ),
            Some(Tensor::from_slice(token_valid).to_kind(Kind::Bool).view([batch, 1])),
            Some(Tensor::zeros([batch, 1, 1, 1], (Kind::Float, device))),
        ));
        DeviceFeaturePacket {
            tensors: TensorPacket {
                matrix: Tensor::zeros([batch, 1, 1], (Kind::Float, device)),
                valid_nodes: Tensor::ones([batch, 1], (Kind::Bool, device)),
                entries,
            },
            schema_id: "winner-activity-test".into(),
            policy_id: "winner-activity-test".into(),
        }
    }

    #[test]
    fn winner_activity_ignores_group_tokens_from_losing_candidate() {
        let chunk = packet(&[0, 1]);
        assert_eq!(chunk.group_activity().int64_value(&[0]), 1);

        let (_, activity) = chunk.freeze_winner_with_activity(
            &Tensor::from_slice(&[0i64]),
            None,
        );
        assert_eq!(activity.int64_value(&[0]), 0);
    }

    #[test]
    fn merged_winner_activity_tracks_the_selected_packet() {
        let chunk = packet(&[0, 1]);
        let incumbent = packet(&[1]);
        let keep_incumbent = Tensor::ones([], (Kind::Bool, Device::Cpu));
        let (winner, activity) = chunk.freeze_winner_with_activity(
            &Tensor::from_slice(&[0i64]),
            Some((&incumbent, &keep_incumbent)),
        );
        assert_eq!(activity.int64_value(&[0]), 1);
        assert_eq!(
            winner.tensors.entries[FeatureId::new(b'B', 1).unwrap().index()]
                .as_ref().unwrap().token_valid.as_ref().unwrap().int64_value(&[0, 0]),
            1,
        );
    }
}
