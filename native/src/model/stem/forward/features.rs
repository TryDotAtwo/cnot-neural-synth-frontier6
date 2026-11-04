use super::super::StructuralStem;
use crate::model::{shape_i64, ModelError};
use crate::packet::FeaturePacket;
use crate::schema::FeatureRole;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

pub(super) struct FeatureAccumulation {
    pub(super) wire: Tensor,
    pub(super) pair: Tensor,
    pub(super) context_tokens: Vec<Tensor>,
    pub(super) context_masks: Vec<Tensor>,
    pub(super) family_available: HashMap<char, Tensor>,
}

pub(super) struct FeatureAccumulationContext<'a> {
    pub(super) device: Device,
    pub(super) batch: i64,
    pub(super) nodes: i64,
    pub(super) valid: &'a Tensor,
    pub(super) pair_mask: &'a Tensor,
}

pub(super) fn accumulate(
    stem: &StructuralStem,
    packet: &FeaturePacket,
    context: FeatureAccumulationContext<'_>,
    mut wire: Tensor,
    mut pair: Tensor,
) -> Result<FeatureAccumulation, ModelError> {
    let FeatureAccumulationContext {
        device,
        batch,
        nodes,
        valid,
        pair_mask,
    } = context;
    let mut context_tokens = Vec::new();
    let mut context_masks = Vec::new();
    let mut family_available: HashMap<char, Tensor> = HashMap::new();

    for spec in &stem.specs {
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
            stem.projections
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
                pair += projected * sample_gate.view([batch, 1, 1, 1]) * pair_mask.unsqueeze(-1);
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
                let role_layers = stem
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

    Ok(FeatureAccumulation {
        wire,
        pair,
        context_tokens,
        context_masks,
        family_available,
    })
}
