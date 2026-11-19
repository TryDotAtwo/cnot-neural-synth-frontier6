//! Shared tensor representation; host-reference conversion is isolated here.
use super::{matrix_batch_values, shape_i64, DeviceFeatureEntry, DeviceFeatureStatus, ModelError};
use crate::packet::{FeaturePacket, FeatureStatus};
use crate::schema::{FeatureSpec, FEATURE_COUNT};
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
pub(super) struct TensorPacket {
    pub(super) matrix: Tensor,
    pub(super) valid_nodes: Tensor,
    pub(super) entries: Vec<Option<DeviceFeatureEntry>>,
}

impl TensorPacket {
    /// Reference/legacy adapter only. The CUDA entry point never calls this.
    pub(super) fn from_host(
        packet: &FeaturePacket,
        specs: &[FeatureSpec],
        device: Device,
    ) -> Result<Self, ModelError> {
        let (batch, nodes) = (packet.batch as i64, packet.nodes as i64);
        let matrix = Tensor::from_slice(&matrix_batch_values(packet)?)
            .view([batch, nodes, nodes])
            .to_device(device);
        let valid_nodes = Tensor::from_slice(&packet.valid_nodes)
            .view([batch, nodes])
            .to_kind(Kind::Bool)
            .to_device(device);
        let mut entries: Vec<Option<DeviceFeatureEntry>> =
            (0..FEATURE_COUNT).map(|_| None).collect();
        for spec in specs {
            let entry = packet.feature(spec.id);
            let shape = shape_i64(&entry.shape)?;
            let statuses: Vec<u8> = entry
                .status
                .iter()
                .map(|status| match status {
                    FeatureStatus::NotRequested => DeviceFeatureStatus::NotRequested as u8,
                    FeatureStatus::Computed => DeviceFeatureStatus::Computed as u8,
                    FeatureStatus::Inapplicable => DeviceFeatureStatus::Inapplicable as u8,
                    FeatureStatus::Unsupported => DeviceFeatureStatus::Unsupported as u8,
                    FeatureStatus::ResourceLimit => DeviceFeatureStatus::ResourceLimit as u8,
                })
                .collect();
            let token_valid = entry.token_valid.as_ref().map(|values| {
                Tensor::from_slice(values)
                    .view([batch, shape[1]])
                    .to_kind(Kind::Bool)
                    .to_device(device)
            });
            let incidence = entry.incidence.as_ref().map(|values| {
                Tensor::from_slice(values)
                    .view([batch, shape[1], nodes, spec.wire_roles.len() as i64])
                    .to_device(device)
            });
            entries[spec.id.index()] = Some(DeviceFeatureEntry::new(
                spec.clone(),
                Tensor::from_slice(&entry.values)
                    .view(shape.as_slice())
                    .to_device(device),
                Tensor::from_slice(&statuses).to_device(device),
                token_valid,
                incidence,
            ));
        }
        Ok(Self {
            matrix,
            valid_nodes,
            entries,
        })
    }
}
