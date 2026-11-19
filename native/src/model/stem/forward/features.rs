use super::super::StructuralStem;
use crate::model::tensor_packet::TensorPacket;
use crate::model::{DeviceFeatureStatus, ModelError};
use crate::schema::FeatureRole;
use std::collections::HashMap;
use tch::{Kind, Tensor};

pub(super) struct FeatureAccumulation {
    pub(super) wire: Tensor,
    pub(super) pair: Tensor,
    pub(super) context_tokens: Vec<Tensor>,
    pub(super) context_masks: Vec<Tensor>,
    pub(super) statuses: Tensor,
    pub(super) family_available: HashMap<char, Tensor>,
}

pub(super) struct FeatureAccumulationContext<'a> {
    pub(super) batch: i64,
    pub(super) valid: &'a Tensor,
    pub(super) pair_mask: &'a Tensor,
}

pub(super) fn accumulate(
    stem: &StructuralStem,
    packet: &TensorPacket,
    context: FeatureAccumulationContext<'_>,
    mut wire: Tensor,
    mut pair: Tensor,
) -> Result<FeatureAccumulation, ModelError> {
    let FeatureAccumulationContext {
        batch,
        valid,
        pair_mask,
    } = context;
    let mut context_tokens = Vec::new();
    let mut context_masks = Vec::new();
    let entries = stem
        .specs
        .iter()
        .map(|spec| {
            let entry = packet.entries[spec.id.index()]
                .as_ref()
                .ok_or_else(|| ModelError(format!("{} tensor input is missing", spec.id)))?;
            if &entry.spec != spec {
                return Err(ModelError(format!(
                    "{} provider schema differs from model schema",
                    spec.id
                )));
            }
            Ok(entry)
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    let statuses = Tensor::stack(&entries.iter().map(|e| &e.status).collect::<Vec<_>>(), 1);
    let computed = statuses.eq(DeviceFeatureStatus::Computed as i64);
    let missing = stem
        .missing_projection
        .as_ref()
        .map(|_| computed.logical_not());
    let gates = computed.to_kind(Kind::Float);
    let valid_float = valid.to_kind(Kind::Float);
    let mut by_family: HashMap<char, Vec<Tensor>> = HashMap::new();
    for (i, (spec, entry)) in stem.specs.iter().zip(entries).enumerate() {
        let shape = entry.values.size();
        let missing_row = missing.as_ref().map(|m| m.select(1, i as i64));
        // Mask before the linear layer: NaN * 0 would also poison its weight gradient.
        let sanitize = |tensor: &Tensor| match &missing_row {
            Some(mask) => {
                let mut broadcast = vec![1; tensor.dim()];
                broadcast[0] = batch;
                tensor.masked_fill(&mask.view(broadcast.as_slice()), 0.0)
            }
            None => tensor.shallow_clone(),
        };
        let projected = sanitize(&entry.values).apply(
            stem.projections
                .get(&spec.id)
                .expect("all model specs are projected"),
        );
        let available = computed.select(1, i as i64);
        by_family
            .entry(spec.id.family())
            .or_default()
            .push(available.shallow_clone());
        let sample_gate = gates.select(1, i as i64);

        match spec.role {
            FeatureRole::Node => {
                wire += projected * sample_gate.view([batch, 1, 1]) * valid_float.unsqueeze(-1);
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
                let token_valid = entry
                    .token_valid
                    .as_ref()
                    .expect("group packet requires token mask")
                    .to_kind(Kind::Bool)
                    .logical_and(&available.unsqueeze(1));
                let group_gate = token_valid.to_kind(Kind::Float);
                let projected = projected * group_gate.unsqueeze(-1);
                let incidence = entry
                    .incidence
                    .as_ref()
                    .expect("group packet requires incidence");
                let incidence = sanitize(incidence);
                let role_layers = stem
                    .group_role_projections
                    .get(&spec.id)
                    .expect("group schema roles are projected");
                for (role_index, role_layer) in role_layers.iter().enumerate() {
                    let attach = incidence.select(-1, role_index as i64)
                        * group_gate.unsqueeze(-1)
                        * valid_float.unsqueeze(1);
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
        statuses,
        wire,
        pair,
        context_tokens,
        context_masks,
        family_available: by_family
            .into_iter()
            .map(|(family, masks)| (family, Tensor::stack(&masks, 1).any_dim(1, false)))
            .collect(),
    })
}
