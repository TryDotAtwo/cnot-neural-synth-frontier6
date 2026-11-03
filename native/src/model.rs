//! Rust implementation for the canonical 103-ID CNOT feature manifest.
//!
//! The distance field consumes 102 registered model inputs; H07 is evaluated
//! as a separate post-forward diagnostic, not as an input tensor.
//! Tensor/autograd operations use libtorch through `tch`; the stable model
//! structure and feature contracts remain owned by this crate. CUDA execution
//! is selected by the tensor device, while `crate::cuda` contains project-owned
//! kernels for exact GF(2) state operations.

use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::packet::{FeaturePacket, FeaturePacketError};
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID};
use std::fmt;
use tch::nn::{self, Module};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

impl From<FeaturePacketError> for ModelError {
    fn from(error: FeaturePacketError) -> Self {
        Self(error.0)
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub wire_dim: i64,
    pub pair_dim: i64,
    pub structural_blocks: usize,
    pub attention_heads: i64,
    pub feedforward_dim: i64,
    pub fusion_blocks: usize,
    pub basis_proposal_hidden_dim: i64,
}

pub(super) const AUXILIARY_HIDDEN_DIM: i64 = 128;

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            wire_dim: 128,
            pair_dim: 32,
            structural_blocks: 6,
            attention_heads: 4,
            feedforward_dim: 256,
            fusion_blocks: 2,
            basis_proposal_hidden_dim: 128,
        }
    }
}

impl ModelConfig {
    fn validate(&self) -> Result<(), ModelError> {
        if self.wire_dim < 1
            || self.pair_dim < 1
            || self.attention_heads < 1
            || self.feedforward_dim < 1
            || self.basis_proposal_hidden_dim < 1
            || self.wire_dim % self.attention_heads != 0
        {
            return Err(ModelError(
                "invalid model dimensions or attention head count".into(),
            ));
        }
        Ok(())
    }
}

fn xavier_uniform(input: i64, output: i64) -> nn::Init {
    let bound = (6.0 / (input + output) as f64).sqrt();
    nn::Init::Uniform {
        lo: -bound,
        up: bound,
    }
}

pub(super) fn pytorch_linear_config(input: i64) -> nn::LinearConfig {
    // torch.nn.Linear.reset_parameters uses kaiming_uniform_(a=sqrt(5)),
    // whose effective uniform bound is 1/sqrt(fan_in).
    let bound = 1.0 / (input as f64).sqrt();
    nn::LinearConfig {
        ws_init: nn::Init::Uniform {
            lo: -bound,
            up: bound,
        },
        ..Default::default()
    }
}

#[derive(Debug)]
pub struct FieldOutput {
    pub value: Tensor,
    pub proposal_logits: Tensor,
    pub qhat: Tensor,
    pub action_mask: Tensor,
}

#[derive(Debug)]
pub(super) struct Mlp {
    first: nn::Linear,
    second: nn::Linear,
}

impl Mlp {
    pub(super) fn new(path: &nn::Path<'_>, input: i64, hidden: i64, output: i64) -> Self {
        Self {
            first: nn::linear(path / "0", input, hidden, pytorch_linear_config(input)),
            second: nn::linear(path / "2", hidden, output, pytorch_linear_config(hidden)),
        }
    }

    pub(super) fn forward(&self, input: &Tensor) -> Tensor {
        input.apply(&self.first).gelu("none").apply(&self.second)
    }
}

#[derive(Debug)]
pub(super) struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: nn::Linear,
    heads: i64,
    width: i64,
}

impl MultiHeadAttention {
    pub(super) fn new(path: &nn::Path<'_>, width: i64, heads: i64) -> Self {
        let out_proj_config = nn::LinearConfig {
            bs_init: Some(nn::Init::Const(0.0)),
            ..pytorch_linear_config(width)
        };
        Self {
            // PyTorch MultiheadAttention stores the three input projections as
            // one [3 * width, width] parameter and names it `in_proj_weight`.
            // Keeping that exact layout makes reference state_dict conversion
            // a byte-for-byte tensor mapping instead of a semantic guess.
            in_proj_weight: path.var(
                "in_proj_weight",
                &[3 * width, width],
                xavier_uniform(width, 3 * width),
            ),
            in_proj_bias: path.var("in_proj_bias", &[3 * width], nn::Init::Const(0.0)),
            out_proj: nn::linear(path / "out_proj", width, width, out_proj_config),
            heads,
            width,
        }
    }

    pub(super) fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_valid: &Tensor,
        pair_bias: Option<&Tensor>,
    ) -> Tensor {
        let query_shape = query.size();
        let batch = query_shape[0];
        let query_len = query_shape[1];
        let key_len = key.size()[1];
        let head_dim = self.width / self.heads;
        let split = |tensor: Tensor, length: i64| {
            tensor
                .view([batch, length, self.heads, head_dim])
                .transpose(1, 2)
        };
        let q_weight = self.in_proj_weight.narrow(0, 0, self.width);
        let k_weight = self.in_proj_weight.narrow(0, self.width, self.width);
        let v_weight = self.in_proj_weight.narrow(0, 2 * self.width, self.width);
        let q_bias = self.in_proj_bias.narrow(0, 0, self.width);
        let k_bias = self.in_proj_bias.narrow(0, self.width, self.width);
        let v_bias = self.in_proj_bias.narrow(0, 2 * self.width, self.width);
        let q = split(query.matmul(&q_weight.transpose(0, 1)) + q_bias, query_len);
        let k = split(key.matmul(&k_weight.transpose(0, 1)) + k_bias, key_len);
        let v = split(value.matmul(&v_weight.transpose(0, 1)) + v_bias, key_len);
        let mut logits = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        if let Some(bias) = pair_bias {
            logits += bias;
        }
        let masked = key_valid.logical_not().unsqueeze(1).unsqueeze(1);
        let weights = logits
            .masked_fill(&masked, f64::NEG_INFINITY)
            .softmax(-1, Kind::Float);
        let attended = weights
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([batch, query_len, self.width]);
        attended.apply(&self.out_proj)
    }
}

#[derive(Debug)]
pub(super) struct StructuralBlock {
    pair_update: Mlp,
    pair_norm: nn::LayerNorm,
    qkv: nn::Linear,
    pair_bias: nn::Linear,
    attention_out: nn::Linear,
    relations: nn::Linear,
    wire_norm: nn::LayerNorm,
    feedforward: Mlp,
    feedforward_norm: nn::LayerNorm,
    wire_dim: i64,
    heads: i64,
}

impl StructuralBlock {
    pub(super) fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
        let d = config.wire_dim;
        let p = config.pair_dim;
        Self {
            pair_update: Mlp::new(
                &(path / "pair_update"),
                2 * p + 2 * d,
                config.feedforward_dim,
                p,
            ),
            pair_norm: nn::layer_norm(path / "pair_norm", vec![p], Default::default()),
            qkv: nn::linear(path / "qkv", d, 3 * d, pytorch_linear_config(d)),
            pair_bias: nn::linear(
                path / "bias",
                p,
                config.attention_heads,
                pytorch_linear_config(p),
            ),
            attention_out: nn::linear(path / "out", d, d, pytorch_linear_config(d)),
            relations: nn::linear(path / "relations", 2 * p, d, pytorch_linear_config(2 * p)),
            wire_norm: nn::layer_norm(path / "norm", vec![d], Default::default()),
            feedforward: Mlp::new(&(path / "ff"), d, config.feedforward_dim, d),
            feedforward_norm: nn::layer_norm(path / "ff_norm", vec![d], Default::default()),
            wire_dim: d,
            heads: config.attention_heads,
        }
    }

    pub(super) fn forward(&self, wire: &Tensor, pair: &Tensor, valid: &Tensor) -> (Tensor, Tensor) {
        let shape = wire.size();
        let batch = shape[0];
        let nodes = shape[1];
        let float_mask = valid.to_kind(wire.kind());
        let pair_mask = float_mask.unsqueeze(2) * float_mask.unsqueeze(1);

        let left = wire
            .unsqueeze(2)
            .expand([batch, nodes, nodes, self.wire_dim], true);
        let right = wire
            .unsqueeze(1)
            .expand([batch, nodes, nodes, self.wire_dim], true);
        let pair_input = Tensor::cat(&[pair, &pair.transpose(1, 2), &left, &right], -1);
        let updated_pair = self
            .pair_norm
            .forward(&(pair + self.pair_update.forward(&pair_input)))
            * pair_mask.unsqueeze(-1);

        let qkv = wire.apply(&self.qkv).chunk(3, -1);
        let head_dim = self.wire_dim / self.heads;
        let split = |tensor: &Tensor| {
            tensor
                .view([batch, nodes, self.heads, head_dim])
                .transpose(1, 2)
        };
        let q = split(&qkv[0]);
        let k = split(&qkv[1]);
        let v = split(&qkv[2]);
        let bias = updated_pair.apply(&self.pair_bias).permute([0, 3, 1, 2]);
        let key_mask = valid.logical_not().unsqueeze(1).unsqueeze(1);
        let logits = (q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt() + bias)
            .masked_fill(&key_mask, f64::NEG_INFINITY);
        let message = logits
            .softmax(-1, Kind::Float)
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([batch, nodes, self.wire_dim]);

        let count = float_mask
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            .clamp_min(1.0)
            .view([batch, 1, 1]);
        let relation_input = Tensor::cat(
            &[
                &updated_pair.sum_dim_intlist([1i64].as_slice(), false, Kind::Float),
                &updated_pair.sum_dim_intlist([2i64].as_slice(), false, Kind::Float),
            ],
            -1,
        ) / count;
        let next_wire = self.wire_norm.forward(
            &(wire + message.apply(&self.attention_out) + relation_input.apply(&self.relations)),
        ) * float_mask.unsqueeze(-1);
        let next_wire = self
            .feedforward_norm
            .forward(&(&next_wire + self.feedforward.forward(&next_wire)))
            * float_mask.unsqueeze(-1);
        (next_wire, updated_pair)
    }
}

mod auxiliary;
mod family_heads;
mod fusion;
mod stem;

use auxiliary::AuxiliaryHeads;
use family_heads::FamilyHeads;
use fusion::FusionTransformer;
pub(crate) use stem::EncodedState;
use stem::StructuralStem;

/// The selected-feature structural distance field used by research_v3.
///
/// `specs` is the selected subset of the canonical 103-ID manifest. Built-in
/// shapes are fixed; a reserved ID receives its role and channel count from
/// its registered provider. The packet still carries all 103 manifest rows.
/// The model rejects packets whose selected IDs, producer metadata, or policy
/// differ from the schema selected at construction. Row availability is an
/// input mask: unavailable rows stay zero-masked and do not block other states
/// in a batch. This v24 contract is distinct from the legacy fixed-shape v22
/// packet adapter.
#[derive(Debug)]
pub struct FusionDistanceField {
    specs: Vec<FeatureSpec>,
    expected_schema_id: String,
    expected_policy_id: String,
    device: Device,
    stem: StructuralStem,
    family_heads: FamilyHeads,
    fusion: FusionTransformer,
    auxiliary: AuxiliaryHeads,
}

impl FusionDistanceField {
    /// Constructs a model directly from the registry and feature policy used
    /// to build its packets, so the model contract cannot drift from either.
    pub fn new_for_policy(
        path: &nn::Path<'_>,
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        let specs = registry
            .selected_specs(policy)
            .map_err(|error| ModelError(error.0))?;
        let policy_id = policy.id().map_err(|error| ModelError(error.0))?;
        Self::new(
            path,
            &specs,
            policy.schema_id.clone(),
            policy_id,
            device,
            config,
        )
    }

    pub fn new(
        path: &nn::Path<'_>,
        specs: &[FeatureSpec],
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        crate::torch_device::require_device(device).map_err(ModelError)?;
        config.validate()?;
        let schema_id = schema_id.into();
        let policy_id = policy_id.into();
        if schema_id.trim().is_empty() || policy_id.trim().is_empty() {
            return Err(ModelError(
                "model requires explicit schema and policy identities".into(),
            ));
        }
        if specs.is_empty() {
            return Err(ModelError(
                "model requires at least one selected feature spec".into(),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(specs.len());
        let mut families: Vec<_> = specs.iter().map(|spec| spec.id.family()).collect();
        for spec in specs {
            if !seen.insert(spec.id) {
                return Err(ModelError(format!(
                    "{} appears more than once in the model feature schema",
                    spec.id
                )));
            }
            if spec.channels == 0 || i64::try_from(spec.channels).is_err() {
                return Err(ModelError(format!(
                    "{} has an invalid channel count",
                    spec.id
                )));
            }
            if spec.producer_version.trim().is_empty()
                || spec.producer_version == "unconfigured"
                || spec.scope.trim().is_empty()
                || spec.exactness.trim().is_empty()
                || spec.exactness == "unknown"
                || spec.cost_tier.trim().is_empty()
                || spec
                    .dependencies
                    .iter()
                    .any(|value| value.trim().is_empty())
            {
                return Err(ModelError(format!(
                    "{} does not have an accepted producer contract",
                    spec.id
                )));
            }
            if spec.role == FeatureRole::Group
                && (spec.wire_roles.is_empty()
                    || spec.wire_roles.iter().any(|role| role.trim().is_empty())
                    || spec
                        .wire_roles
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != spec.wire_roles.len())
            {
                return Err(ModelError(format!(
                    "{} requires a unique physical wire-role layout",
                    spec.id
                )));
            }
            if spec.role != FeatureRole::Group && !spec.wire_roles.is_empty() {
                return Err(ModelError(format!(
                    "{} has group roles on a non-group feature",
                    spec.id
                )));
            }
        }
        families.sort_unstable();
        families.dedup();

        Ok(Self {
            specs: specs.to_vec(),
            expected_schema_id: schema_id,
            expected_policy_id: policy_id,
            device,
            stem: StructuralStem::new(&(path / "stem"), specs, &config),
            family_heads: FamilyHeads::new(&(path / "family_heads"), &families, &config),
            fusion: FusionTransformer::new(&(path / "fusion"), &config, families.len()),
            auxiliary: AuxiliaryHeads::new(&(path / "auxiliary"), &config),
        })
    }

    pub(crate) fn encode(&self, packet: &FeaturePacket) -> Result<EncodedState, ModelError> {
        if packet.schema_id != self.expected_schema_id
            || packet.policy_id != self.expected_policy_id
        {
            return Err(ModelError(
                "feature packet schema or policy differs from model contract".into(),
            ));
        }
        if packet.feature_id_manifest_id != FEATURE_ID_MANIFEST_ID
            || packet.entries.len() != FEATURE_COUNT
            || packet
                .entries
                .iter()
                .enumerate()
                .any(|(index, entry)| FeatureId::from_index(index) != Some(entry.spec.id))
        {
            return Err(ModelError(format!(
                "feature packet must preserve the canonical {FEATURE_COUNT}-ID manifest"
            )));
        }
        let model_ids: std::collections::HashSet<_> =
            self.specs.iter().map(|spec| spec.id).collect();
        let packet_ids: std::collections::HashSet<_> = packet
            .entries
            .iter()
            .filter(|entry| {
                entry
                    .status
                    .iter()
                    .any(|status| *status != crate::packet::FeatureStatus::NotRequested)
            })
            .map(|entry| entry.spec.id)
            .collect();
        if model_ids != packet_ids {
            return Err(ModelError(
                "selected packet feature IDs differ from model input schema".into(),
            ));
        }
        if self.specs.iter().any(|spec| {
            packet
                .entries
                .get(spec.id.index())
                .is_none_or(|entry| entry.spec != *spec)
        }) {
            return Err(ModelError(
                "feature packet producer schema differs from model contract".into(),
            ));
        }
        self.stem.forward(packet, self.device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub(crate) fn forward_encoded(&self, encoded: &EncodedState) -> FieldOutput {
        let (family_tokens, family_masks) = self.family_heads.forward(&encoded);
        let positive_value = self.fusion.forward(&encoded, &family_tokens, &family_masks);
        let value =
            Tensor::zeros_like(&positive_value).where_self(&encoded.identity, &positive_value);
        let (proposal_logits, qhat, action_mask) =
            self.auxiliary
                .forward(&encoded.wire, &encoded.pair, &value, &encoded.valid);
        FieldOutput {
            value,
            proposal_logits,
            qhat,
            action_mask,
        }
    }

    pub fn forward(&self, packet: &FeaturePacket) -> Result<FieldOutput, ModelError> {
        let encoded = self.encode(packet)?;
        Ok(self.forward_encoded(&encoded))
    }
}

fn shape_i64(shape: &[usize]) -> Result<Vec<i64>, ModelError> {
    shape
        .iter()
        .map(|extent| {
            i64::try_from(*extent)
                .map_err(|_| ModelError("feature shape exceeds tensor index range".into()))
        })
        .collect()
}

fn matrix_batch_values(packet: &FeaturePacket) -> Result<Vec<f32>, ModelError> {
    let total = packet
        .batch
        .checked_mul(packet.nodes)
        .and_then(|value| value.checked_mul(packet.nodes))
        .ok_or_else(|| ModelError("matrix batch size overflow".into()))?;
    let mut values = vec![0.0f32; total];
    for (batch_index, state) in packet.state_keys.iter().enumerate() {
        let dimension = usize::try_from(state.dimension)
            .map_err(|_| ModelError("matrix dimension exceeds host index range".into()))?;
        let words_per_row = dimension
            .checked_add(63)
            .ok_or_else(|| ModelError("matrix row width overflow".into()))?
            / 64;
        for row in 0..dimension {
            for column in 0..dimension {
                let word = state.packed_rows[row * words_per_row + column / 64];
                if (word >> (column % 64)) & 1 != 0 {
                    values[(batch_index * packet.nodes + row) * packet.nodes + column] = 1.0;
                }
            }
        }
    }
    Ok(values)
}
