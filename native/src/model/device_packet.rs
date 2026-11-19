//! CUDA-resident model inputs. Validation reads tensor metadata, never payloads.
//!
//! Producers own numerical validity and the meaning of each status. A selected
//! but unavailable feature must still supply its typed entry and device status;
//! an absent selected entry is an integration error, not a computed zero.
#[cfg(feature = "cuda")]
mod frozen;
#[cfg(feature = "cuda")]
mod layout;

use super::tensor_packet::TensorPacket;
use super::ModelError;
use crate::schema::{
    FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID,
    H07_POST_FORWARD_DIAGNOSTIC_ID,
};
use tch::{Device, Kind, Tensor};

/// Stable device ABI, independent of the host enum's Rust discriminants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DeviceFeatureStatus {
    NotRequested = 0,
    Computed = 1,
    Inapplicable = 2,
    Unsupported = 3,
    ResourceLimit = 4,
}

#[derive(Debug)]
pub struct DeviceFeatureEntry {
    pub(super) spec: FeatureSpec,
    pub(super) values: Tensor,
    pub(super) status: Tensor,
    pub(super) token_valid: Option<Tensor>,
    pub(super) incidence: Option<Tensor>,
}

impl DeviceFeatureEntry {
    /// Shapes and devices are validated when the entry joins a packet.
    pub fn new(
        spec: FeatureSpec,
        values: Tensor,
        status: Tensor,
        token_valid: Option<Tensor>,
        incidence: Option<Tensor>,
    ) -> Self {
        Self {
            spec,
            values,
            status,
            token_valid,
            incidence,
        }
    }

    pub fn spec(&self) -> &FeatureSpec {
        &self.spec
    }
    pub fn values(&self) -> &Tensor {
        &self.values
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    pub fn token_valid(&self) -> Option<&Tensor> {
        self.token_valid.as_ref()
    }
    pub fn incidence(&self) -> Option<&Tensor> {
        self.incidence.as_ref()
    }
}

#[derive(Debug)]
pub struct DeviceFeaturePacket {
    pub(super) tensors: TensorPacket,
    pub(super) schema_id: String,
    pub(super) policy_id: String,
}

impl DeviceFeaturePacket {
    /// Matrix is dense f32 [batch,n,n]; valid_nodes is bool [batch,n]. Both
    /// must already be on CUDA. The constructor neither uploads nor downloads.
    /// Entries retain handles to producer storage on the current Torch stream.
    pub fn new(
        matrix: Tensor,
        valid_nodes: Tensor,
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
        entries: Vec<DeviceFeatureEntry>,
    ) -> Result<Self, ModelError> {
        let device = matrix.device();
        if !matches!(device, Device::Cuda(_)) {
            return Err(ModelError(
                "device model packet requires CUDA tensors".into(),
            ));
        }
        let shape = matrix.size();
        if matrix.kind() != Kind::Float
            || shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[1] != shape[2]
        {
            return Err(ModelError(
                "device packet matrix must be f32 [batch,n,n] with positive extents".into(),
            ));
        }
        let (batch, nodes) = (shape[0], shape[1]);
        check_tensor(
            &valid_nodes,
            device,
            Kind::Bool,
            &[batch, nodes],
            "node mask",
        )?;
        let schema_id = schema_id.into();
        let policy_id = policy_id.into();
        if schema_id.trim().is_empty() || policy_id.trim().is_empty() || entries.is_empty() {
            return Err(ModelError(
                "device packet requires schema, policy and selected feature entries".into(),
            ));
        }
        let mut slots: Vec<Option<DeviceFeatureEntry>> = (0..FEATURE_COUNT).map(|_| None).collect();
        for entry in entries {
            let index = entry.spec.id.index();
            if entry.spec.id == H07_POST_FORWARD_DIAGNOSTIC_ID {
                return Err(ModelError(
                    "H07 is a post-forward diagnostic, not a model input".into(),
                ));
            }
            if slots[index].is_some() {
                return Err(ModelError(format!(
                    "duplicate device feature {}",
                    entry.spec.id
                )));
            }
            validate_entry(&entry, device, batch, nodes)?;
            slots[index] = Some(entry);
        }
        Ok(Self {
            tensors: TensorPacket {
                matrix,
                valid_nodes,
                entries: slots,
            },
            schema_id,
            policy_id,
        })
    }

    pub fn manifest_id(&self) -> &'static str {
        FEATURE_ID_MANIFEST_ID
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn schema_id(&self) -> &str {
        &self.schema_id
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn policy_id(&self) -> &str {
        &self.policy_id
    }
    pub fn feature(&self, id: FeatureId) -> Option<&DeviceFeatureEntry> {
        self.tensors.entries[id.index()].as_ref()
    }
    pub fn matrix(&self) -> &Tensor {
        &self.tensors.matrix
    }
    pub fn valid_nodes(&self) -> &Tensor {
        &self.tensors.valid_nodes
    }

    pub(super) fn validate_model(
        &self,
        specs: &[FeatureSpec],
        schema_id: &str,
        policy_id: &str,
        device: Device,
    ) -> Result<(), ModelError> {
        if self.schema_id != schema_id
            || self.policy_id != policy_id
            || self.matrix().device() != device
        {
            return Err(ModelError(
                "device packet schema, policy or device differs from model contract".into(),
            ));
        }
        if self
            .tensors
            .entries
            .iter()
            .filter(|entry| entry.is_some())
            .count()
            != specs.len()
        {
            return Err(ModelError(
                "selected device feature IDs differ from model input schema".into(),
            ));
        }
        for spec in specs {
            if self
                .feature(spec.id)
                .is_none_or(|entry| entry.spec != *spec)
            {
                return Err(ModelError(format!(
                    "{} device producer is missing or differs from model contract",
                    spec.id
                )));
            }
        }
        Ok(())
    }
}

fn check_tensor(
    tensor: &Tensor,
    device: Device,
    kind: Kind,
    shape: &[i64],
    name: &str,
) -> Result<(), ModelError> {
    if tensor.device() != device || tensor.kind() != kind || tensor.size() != shape {
        return Err(ModelError(format!(
            "{name} tensor device, dtype or shape does not match the device packet"
        )));
    }
    Ok(())
}

fn validate_entry(
    entry: &DeviceFeatureEntry,
    device: Device,
    batch: i64,
    nodes: i64,
) -> Result<(), ModelError> {
    let spec = &entry.spec;
    let channels = i64::try_from(spec.channels)
        .map_err(|_| ModelError("feature channel count exceeds tensor range".into()))?;
    if channels <= 0 {
        return Err(ModelError("feature channel count must be positive".into()));
    }
    check_tensor(
        &entry.status,
        device,
        Kind::Uint8,
        &[batch],
        "feature status",
    )?;
    let shape = entry.values.size();
    let expected = match spec.role {
        FeatureRole::Node => vec![batch, nodes, channels],
        FeatureRole::Pair => vec![batch, nodes, nodes, channels],
        FeatureRole::Global => vec![batch, channels],
        FeatureRole::Group => {
            if shape.len() != 3 || spec.wire_roles.is_empty() {
                return Err(ModelError(format!(
                    "{} group feature requires token and physical-role axes",
                    spec.id
                )));
            }
            let groups = shape[1];
            let token_valid = entry
                .token_valid
                .as_ref()
                .ok_or_else(|| ModelError("group token mask is missing".into()))?;
            if token_valid.kind() != Kind::Bool && token_valid.kind() != Kind::Uint8 {
                return Err(ModelError("group token mask must be bool or u8".into()));
            }
            check_tensor(
                token_valid,
                device,
                token_valid.kind(),
                &[batch, groups],
                "group token mask",
            )?;
            let incidence = entry
                .incidence
                .as_ref()
                .ok_or_else(|| ModelError("group incidence is missing".into()))?;
            check_tensor(
                incidence,
                device,
                Kind::Float,
                &[batch, groups, nodes, spec.wire_roles.len() as i64],
                "group incidence",
            )?;
            vec![batch, groups, channels]
        }
    };
    if spec.role != FeatureRole::Group
        && (entry.token_valid.is_some() || entry.incidence.is_some() || !spec.wire_roles.is_empty())
    {
        return Err(ModelError(
            "non-group feature carries group metadata".into(),
        ));
    }
    check_tensor(
        &entry.values,
        device,
        Kind::Float,
        &expected,
        &spec.id.to_string(),
    )
}
