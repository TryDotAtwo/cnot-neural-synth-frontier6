use super::{EncodedState, StructuralStem};
use crate::model::{matrix_batch_values, shape_i64, ModelError};
use crate::packet::FeaturePacket;
use crate::schema::FeatureRole;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

impl StructuralStem {
    pub(in crate::model) fn forward(
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
