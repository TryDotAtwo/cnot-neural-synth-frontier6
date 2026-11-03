use super::{
    matrix_batch_values, pytorch_linear_config, shape_i64, ModelConfig, ModelError, StructuralBlock,
};
use crate::packet::FeaturePacket;
use crate::schema::{FeatureId, FeatureRole, FeatureSpec};
use std::collections::HashMap;
use tch::nn;
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
pub(super) struct StructuralStem {
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
    pub(super) pair: Tensor,
    pub(super) context: Tensor,
    pub(super) context_valid: Tensor,
    pub(crate) valid: Tensor,
    pub(super) family_available: HashMap<char, Tensor>,
    pub(super) identity: Tensor,
}

impl StructuralStem {
    pub(super) fn new(path: &nn::Path<'_>, specs: &[FeatureSpec], config: &ModelConfig) -> Self {
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

    pub(super) fn forward(
        &self,
        packet: &FeaturePacket,
        device: Device,
    ) -> Result<EncodedState, ModelError> {
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
