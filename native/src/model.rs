//! Rust implementation of the 103-feature distance field.
//!
//! Tensor/autograd operations use libtorch through `tch`; the stable model
//! structure and feature contracts remain owned by this crate. CUDA execution
//! is selected by the tensor device, while `crate::cuda` contains project-owned
//! kernels for exact GF(2) state operations.

use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::packet::{FeaturePacket, FeaturePacketError};
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID};
use std::collections::HashMap;
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

const AUXILIARY_HIDDEN_DIM: i64 = 128;

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

fn pytorch_linear_config(input: i64) -> nn::LinearConfig {
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
struct Mlp {
    first: nn::Linear,
    second: nn::Linear,
}

impl Mlp {
    fn new(path: &nn::Path<'_>, input: i64, hidden: i64, output: i64) -> Self {
        Self {
            first: nn::linear(path / "0", input, hidden, pytorch_linear_config(input)),
            second: nn::linear(path / "2", hidden, output, pytorch_linear_config(hidden)),
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.apply(&self.first).gelu("none").apply(&self.second)
    }
}

#[derive(Debug)]
struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: nn::Linear,
    heads: i64,
    width: i64,
}

impl MultiHeadAttention {
    fn new(path: &nn::Path<'_>, width: i64, heads: i64) -> Self {
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

    fn forward(
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
struct StructuralBlock {
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
    fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
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

    fn forward(&self, wire: &Tensor, pair: &Tensor, valid: &Tensor) -> (Tensor, Tensor) {
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

#[derive(Debug)]
struct StructuralStem {
    specs: Vec<FeatureSpec>,
    projections: HashMap<FeatureId, nn::Linear>,
    group_role_projections: HashMap<FeatureId, Vec<nn::Linear>>,
    raw_matrix: nn::Linear,
    pair_to_wire: nn::Linear,
    size_projection: nn::Linear,
    context_query: nn::Linear,
    context_key: nn::Linear,
    context_value: nn::Linear,
    blocks: Vec<StructuralBlock>,
    wire_dim: i64,
}

#[derive(Debug)]
pub(crate) struct EncodedState {
    pub(crate) wire: Tensor,
    pair: Tensor,
    context: Tensor,
    context_valid: Tensor,
    pub(crate) valid: Tensor,
    family_available: HashMap<char, Tensor>,
    identity: Tensor,
}

impl StructuralStem {
    fn new(path: &nn::Path<'_>, specs: &[FeatureSpec], config: &ModelConfig) -> Self {
        let mut projections = HashMap::with_capacity(specs.len());
        let mut group_role_projections = HashMap::new();
        for spec in specs {
            let name = spec.id.to_string();
            let out_dim = if spec.role == FeatureRole::Pair {
                config.pair_dim
            } else {
                config.wire_dim
            };
            projections.insert(
                spec.id,
                nn::linear(
                    path / "projections" / name.as_str(),
                    spec.channels as i64,
                    out_dim,
                    pytorch_linear_config(spec.channels as i64),
                ),
            );
            if spec.role == FeatureRole::Group {
                let mut role_layers = Vec::with_capacity(spec.wire_roles.len());
                for role_index in 0..spec.wire_roles.len() {
                    role_layers.push(nn::linear(
                        path / "group_roles" / name.as_str() / role_index.to_string().as_str(),
                        config.wire_dim,
                        config.wire_dim,
                        nn::LinearConfig {
                            bias: false,
                            ..pytorch_linear_config(config.wire_dim)
                        },
                    ));
                }
                group_role_projections.insert(spec.id, role_layers);
            }
        }

        Self {
            specs: specs.to_vec(),
            projections,
            group_role_projections,
            raw_matrix: nn::linear(
                path / "raw_matrix",
                3,
                config.pair_dim,
                pytorch_linear_config(3),
            ),
            pair_to_wire: nn::linear(
                path / "pair_to_node",
                2 * config.pair_dim,
                config.wire_dim,
                pytorch_linear_config(2 * config.pair_dim),
            ),
            size_projection: nn::linear(
                path / "size_projection",
                2,
                config.wire_dim,
                pytorch_linear_config(2),
            ),
            context_query: nn::linear(
                path / "context_q",
                config.wire_dim,
                config.wire_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            context_key: nn::linear(
                path / "context_k",
                config.wire_dim,
                config.wire_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            context_value: nn::linear(
                path / "context_v",
                config.wire_dim,
                config.wire_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            blocks: (0..config.structural_blocks)
                .map(|index| {
                    let name = index.to_string();
                    StructuralBlock::new(&(path / "blocks" / name.as_str()), config)
                })
                .collect(),
            wire_dim: config.wire_dim,
        }
    }

    fn forward(&self, packet: &FeaturePacket, device: Device) -> Result<EncodedState, ModelError> {
        let batch = packet.batch as i64;
        let nodes = packet.nodes as i64;
        let matrix_data = matrix_batch_values(packet)?;
        let matrix = Tensor::from_slice(&matrix_data)
            .view([batch, nodes, nodes])
            .to_device(device);
        let valid = Tensor::from_slice(&packet.valid_nodes)
            .view([batch, nodes])
            .to_kind(Kind::Bool)
            .to_device(device);
        let eye = Tensor::eye(nodes, (Kind::Float, device))
            .unsqueeze(0)
            .expand([batch, nodes, nodes], true);
        let pair_mask =
            valid.to_kind(Kind::Float).unsqueeze(2) * valid.to_kind(Kind::Float).unsqueeze(1);
        let matrix_channels = Tensor::stack(&[&matrix, &matrix.transpose(1, 2), &eye], -1);
        let mut pair = matrix_channels.apply(&self.raw_matrix) * pair_mask.unsqueeze(-1);
        let sizes =
            valid
                .to_kind(Kind::Float)
                .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
        let size_features = Tensor::stack(&[&sizes, &sizes.log1p()], -1);
        let mut wire = size_features
            .apply(&self.size_projection)
            .unsqueeze(1)
            .expand([batch, nodes, self.wire_dim], true)
            * valid.to_kind(Kind::Float).unsqueeze(-1);

        let mut context_tokens = Vec::new();
        let mut context_masks = Vec::new();
        let mut family_available: HashMap<char, Tensor> = HashMap::new();

        for spec in &self.specs {
            let entry = packet.feature(spec.id);
            if &entry.spec != spec {
                return Err(ModelError(format!(
                    "{} provider schema differs from model schema",
                    spec.id
                )));
            }
            let shape = shape_i64(&entry.shape)?;
            let raw = Tensor::from_slice(&entry.values)
                .view(shape.as_slice())
                .to_device(device);
            let projected = raw.apply(
                self.projections
                    .get(&spec.id)
                    .expect("all model specs are projected"),
            );
            let available = Tensor::from_slice(&entry.available)
                .to_kind(Kind::Bool)
                .to_device(device);
            if let Some(previous) = family_available.remove(&spec.id.family()) {
                family_available.insert(spec.id.family(), previous.logical_or(&available));
            } else {
                family_available.insert(spec.id.family(), available.shallow_clone());
            }
            let sample_gate = available.to_kind(Kind::Float);

            match spec.role {
                FeatureRole::Node => {
                    wire += projected
                        * sample_gate.view([batch, 1, 1])
                        * valid.to_kind(Kind::Float).unsqueeze(-1);
                }
                FeatureRole::Pair => {
                    pair +=
                        projected * sample_gate.view([batch, 1, 1, 1]) * pair_mask.unsqueeze(-1);
                }
                FeatureRole::Global => {
                    let token = projected * sample_gate.view([batch, 1]);
                    context_tokens.push(token.unsqueeze(1));
                    context_masks.push(available.unsqueeze(1));
                }
                FeatureRole::Group => {
                    let groups = shape[1];
                    if groups == 0 {
                        continue;
                    }
                    let token_valid = Tensor::from_slice(
                        entry
                            .token_valid
                            .as_ref()
                            .expect("group packet requires token mask"),
                    )
                    .view([batch, groups])
                    .to_kind(Kind::Bool)
                    .to_device(device)
                    .logical_and(&available.unsqueeze(1));
                    let group_gate = token_valid.to_kind(Kind::Float);
                    let projected = projected * group_gate.unsqueeze(-1);
                    let incidence = Tensor::from_slice(
                        entry
                            .incidence
                            .as_ref()
                            .expect("group packet requires incidence"),
                    )
                    .view([batch, groups, nodes, spec.wire_roles.len() as i64])
                    .to_device(device);
                    let role_layers = self
                        .group_role_projections
                        .get(&spec.id)
                        .expect("group schema roles are projected");
                    for (role_index, role_layer) in role_layers.iter().enumerate() {
                        let attach = incidence.select(-1, role_index as i64)
                            * group_gate.unsqueeze(-1)
                            * valid.to_kind(Kind::Float).unsqueeze(1);
                        let messages = attach.transpose(1, 2).bmm(&projected.apply(role_layer));
                        let denominator = attach
                            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                            .clamp_min(1.0)
                            .unsqueeze(-1);
                        wire += messages / denominator;
                    }
                    context_tokens.push(projected);
                    context_masks.push(token_valid);
                }
            }
        }

        let count = sizes.clamp_min(1.0).view([batch, 1, 1]);
        let pair_to_wire = Tensor::cat(
            &[
                &pair.sum_dim_intlist([1i64].as_slice(), false, Kind::Float),
                &pair.sum_dim_intlist([2i64].as_slice(), false, Kind::Float),
            ],
            -1,
        ) / count;
        wire += pair_to_wire.apply(&self.pair_to_wire) * valid.to_kind(Kind::Float).unsqueeze(-1);

        let null_token = Tensor::zeros([batch, 1, self.wire_dim], (Kind::Float, device));
        let null_valid = Tensor::ones([batch, 1], (Kind::Bool, device));
        context_tokens.push(null_token);
        context_masks.push(null_valid);
        let context = Tensor::cat(&context_tokens.iter().collect::<Vec<_>>(), 1);
        let context_valid = Tensor::cat(&context_masks.iter().collect::<Vec<_>>(), 1);

        let query = wire.apply(&self.context_query);
        let key = context.apply(&self.context_key);
        let value = context.apply(&self.context_value);
        let logits = query.matmul(&key.transpose(1, 2)) / (self.wire_dim as f64).sqrt();
        let logits =
            logits.masked_fill(&context_valid.logical_not().unsqueeze(1), f64::NEG_INFINITY);
        let message = logits.softmax(-1, Kind::Float).matmul(&value);
        wire = (wire + message) * valid.to_kind(Kind::Float).unsqueeze(-1);

        for block in &self.blocks {
            (wire, pair) = block.forward(&wire, &pair, &valid);
        }
        let identity = ((&matrix - &eye).abs() * pair_mask)
            .sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Float)
            .eq(0.0);
        Ok(EncodedState {
            wire,
            pair,
            context,
            context_valid,
            valid,
            family_available,
            identity,
        })
    }
}

#[derive(Debug)]
struct FamilyHeads {
    heads: HashMap<char, Mlp>,
    order: Vec<char>,
}

impl FamilyHeads {
    fn new(path: &nn::Path<'_>, families: &[char], config: &ModelConfig) -> Self {
        let mut heads = HashMap::new();
        for family in families {
            heads.insert(
                *family,
                Mlp::new(
                    &(path / "heads" / family.to_string().as_str()),
                    config.wire_dim + 2 * config.pair_dim,
                    config.wire_dim,
                    config.wire_dim,
                ),
            );
        }
        Self {
            heads,
            order: families.to_vec(),
        }
    }

    fn forward(&self, encoded: &EncodedState) -> (Tensor, Tensor) {
        let size = encoded
            .valid
            .to_kind(Kind::Float)
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            .clamp_min(1.0)
            .view([-1, 1, 1]);
        let input = Tensor::cat(
            &[
                &encoded.wire,
                &(encoded
                    .pair
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                    / &size),
                &(encoded
                    .pair
                    .sum_dim_intlist([2i64].as_slice(), false, Kind::Float)
                    / &size),
            ],
            -1,
        );
        let mut tokens = Vec::with_capacity(self.order.len());
        let mut masks = Vec::with_capacity(self.order.len());
        for family in &self.order {
            let family_valid = encoded
                .family_available
                .get(family)
                .expect("all 103 feature families are present")
                .unsqueeze(1)
                .expand_as(&encoded.valid)
                .logical_and(&encoded.valid);
            let token = self
                .heads
                .get(family)
                .expect("family head is constructed")
                .forward(&input)
                * family_valid.to_kind(input.kind()).unsqueeze(-1);
            tokens.push(token);
            masks.push(family_valid);
        }
        (
            Tensor::stack(&tokens.iter().collect::<Vec<_>>(), 2),
            Tensor::stack(&masks.iter().collect::<Vec<_>>(), 2),
        )
    }
}

#[derive(Debug)]
struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    attention_norm: nn::LayerNorm,
    first: nn::Linear,
    second: nn::Linear,
    feedforward_norm: nn::LayerNorm,
}

impl TransformerEncoderLayer {
    fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                &(path / "self_attn"),
                config.wire_dim,
                config.attention_heads,
            ),
            attention_norm: nn::layer_norm(
                path / "norm1",
                vec![config.wire_dim],
                Default::default(),
            ),
            first: nn::linear(
                path / "linear1",
                config.wire_dim,
                config.feedforward_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            second: nn::linear(
                path / "linear2",
                config.feedforward_dim,
                config.wire_dim,
                pytorch_linear_config(config.feedforward_dim),
            ),
            feedforward_norm: nn::layer_norm(
                path / "norm2",
                vec![config.wire_dim],
                Default::default(),
            ),
        }
    }

    fn forward(&self, input: &Tensor, valid: &Tensor) -> Tensor {
        let attended = self.attention.forward(input, input, input, valid, None);
        let hidden = self.attention_norm.forward(&(input + attended));
        self.feedforward_norm
            .forward(&(&hidden + hidden.apply(&self.first).relu().apply(&self.second)))
    }
}

#[derive(Debug)]
struct FusionTransformer {
    family_type: Tensor,
    family_encoder: TransformerEncoderLayer,
    structural_blocks: Vec<StructuralBlock>,
    distance_query: Tensor,
    readout: MultiHeadAttention,
    readout_norm: nn::LayerNorm,
    output: Mlp,
    family_count: i64,
    wire_dim: i64,
}

impl FusionTransformer {
    fn new(path: &nn::Path<'_>, config: &ModelConfig, family_count: usize) -> Self {
        let family_type = path.var(
            "family_type",
            &[family_count as i64, config.wire_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let distance_query = path.var(
            "dist",
            &[1, 1, config.wire_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        Self {
            family_type,
            family_encoder: TransformerEncoderLayer::new(&(path / "family_attention"), config),
            structural_blocks: (0..config.fusion_blocks)
                .map(|index| {
                    let name = index.to_string();
                    StructuralBlock::new(&(path / "blocks" / name.as_str()), config)
                })
                .collect(),
            distance_query,
            readout: MultiHeadAttention::new(
                &(path / "read"),
                config.wire_dim,
                config.attention_heads,
            ),
            readout_norm: nn::layer_norm(path / "norm", vec![config.wire_dim], Default::default()),
            output: Mlp::new(
                &(path / "output"),
                config.wire_dim + 2,
                config.feedforward_dim,
                1,
            ),
            family_count: family_count as i64,
            wire_dim: config.wire_dim,
        }
    }

    fn forward(
        &self,
        encoded: &EncodedState,
        family_tokens: &Tensor,
        available: &Tensor,
    ) -> Tensor {
        let shape = family_tokens.size();
        let batch = shape[0];
        let nodes = shape[1];
        let typed = (family_tokens
            + self
                .family_type
                .view([1, 1, self.family_count, self.wire_dim]))
            * available.to_kind(family_tokens.kind()).unsqueeze(-1);
        let null_family = Tensor::zeros(
            [batch, nodes, 1, self.wire_dim],
            (family_tokens.kind(), family_tokens.device()),
        );
        let family_inputs = Tensor::cat(&[&typed, &null_family], 2).view([
            batch * nodes,
            self.family_count + 1,
            self.wire_dim,
        ]);
        let null_valid = Tensor::ones([batch, nodes, 1], (Kind::Bool, available.device()));
        let family_mask =
            Tensor::cat(&[available, &null_valid], 2).view([batch * nodes, self.family_count + 1]);
        let attended = self.family_encoder.forward(&family_inputs, &family_mask);
        let family_count = family_mask
            .to_kind(attended.kind())
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            .clamp_min(1.0)
            .unsqueeze(-1);
        let pooled = (attended * family_mask.to_kind(family_inputs.kind()).unsqueeze(-1))
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            / family_count;
        let mut wire = (encoded.wire.shallow_clone() + pooled.view([batch, nodes, self.wire_dim]))
            * encoded.valid.to_kind(family_inputs.kind()).unsqueeze(-1);
        let mut pair = encoded.pair.shallow_clone();
        for block in &self.structural_blocks {
            (wire, pair) = block.forward(&wire, &pair, &encoded.valid);
        }
        let context = Tensor::cat(&[&wire, &encoded.context], 1);
        let context_valid = Tensor::cat(&[&encoded.valid, &encoded.context_valid], 1);
        let query = self.distance_query.expand([batch, 1, self.wire_dim], true);
        let read = self
            .readout
            .forward(&query, &context, &context, &context_valid, None);
        let summary = self.readout_norm.forward(&(query + read)).select(1, 0);
        let size = encoded.valid.to_kind(summary.kind()).sum_dim_intlist(
            [1i64].as_slice(),
            false,
            Kind::Float,
        );
        self.output
            .forward(&Tensor::cat(
                &[&summary, &size.unsqueeze(-1), &size.log1p().unsqueeze(-1)],
                -1,
            ))
            .squeeze_dim(-1)
            .softplus()
    }
}

#[derive(Debug)]
struct AuxiliaryHeads {
    proposal: Mlp,
    qhat: Mlp,
    wire_dim: i64,
}

impl AuxiliaryHeads {
    fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
        let input = 2 * config.wire_dim + 2 * config.pair_dim + 1;
        Self {
            proposal: Mlp::new(&(path / "proposal"), input, AUXILIARY_HIDDEN_DIM, 1),
            qhat: Mlp::new(&(path / "qhat"), input, AUXILIARY_HIDDEN_DIM, 1),
            wire_dim: config.wire_dim,
        }
    }

    fn forward(
        &self,
        wire: &Tensor,
        pair: &Tensor,
        value: &Tensor,
        valid: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let shape = wire.size();
        let batch = shape[0];
        let nodes = shape[1];
        let control = wire
            .unsqueeze(2)
            .expand([batch, nodes, nodes, self.wire_dim], true);
        let target = wire
            .unsqueeze(1)
            .expand([batch, nodes, nodes, self.wire_dim], true);
        let shared_value = value
            .detach()
            .view([batch, 1, 1, 1])
            .expand([batch, nodes, nodes, 1], true);
        let input = Tensor::cat(
            &[
                &control,
                &target,
                pair,
                &pair.transpose(1, 2),
                &shared_value,
            ],
            -1,
        );
        let mut action_mask = valid.unsqueeze(2).logical_and(&valid.unsqueeze(1));
        let eye = Tensor::eye(nodes, (Kind::Bool, valid.device()));
        action_mask = action_mask.logical_and(&eye.logical_not().unsqueeze(0));
        let logits = self
            .proposal
            .forward(&input)
            .squeeze_dim(-1)
            .masked_fill(&action_mask.logical_not(), f64::NEG_INFINITY);
        let qhat = (self.qhat.forward(&input).squeeze_dim(-1).softplus() + 1.0)
            .masked_fill(&action_mask.logical_not(), f64::INFINITY);
        (logits, qhat, action_mask)
    }
}

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
        let value = self.fusion.forward(&encoded, &family_tokens, &family_masks)
            * encoded.identity.logical_not().to_kind(Kind::Float);
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
