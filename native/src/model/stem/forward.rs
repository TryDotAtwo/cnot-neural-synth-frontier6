use super::{EncodedState, StructuralStem};
use crate::model::tensor_packet::TensorPacket;
use crate::model::ModelError;
use tch::{Kind, Tensor};

mod features;
use features::FeatureAccumulation;

impl StructuralStem {
    pub(in crate::model) fn forward(
        &self,
        packet: &TensorPacket,
        share_wire_projections: bool,
    ) -> Result<EncodedState, ModelError> {
        let batch = packet.matrix.size()[0];
        let nodes = packet.matrix.size()[1];
        let device = packet.matrix.device();
        let matrix = packet.matrix.shallow_clone();
        let valid = packet.valid_nodes.shallow_clone();
        let eye = Tensor::eye(nodes, (Kind::Float, device))
            .unsqueeze(0)
            .expand([batch, nodes, nodes], true);
        let pair_mask =
            valid.to_kind(Kind::Float).unsqueeze(2) * valid.to_kind(Kind::Float).unsqueeze(1);
        let matrix_channels = Tensor::stack(&[&matrix, &matrix.transpose(1, 2), &eye], -1);
        let pair = matrix_channels.apply(&self.raw_matrix) * pair_mask.unsqueeze(-1);
        let sizes =
            valid
                .to_kind(Kind::Float)
                .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
        let size_features = Tensor::stack(&[&sizes, &sizes.log1p()], -1);
        let wire = size_features
            .apply(&self.size_projection)
            .unsqueeze(1)
            .expand([batch, nodes, self.wire_dim], true)
            * valid.to_kind(Kind::Float).unsqueeze(-1);

        let FeatureAccumulation {
            mut wire,
            mut pair,
            mut context_tokens,
            mut context_masks,
            statuses,
            family_available,
        } = features::accumulate(
            self,
            packet,
            features::FeatureAccumulationContext {
                batch,
                valid: &valid,
                pair_mask: &pair_mask,
            },
            wire,
            pair,
        )?;

        let count = sizes.clamp_min(1.0).view([batch, 1, 1]);
        let pair_to_wire = Tensor::cat(
            &[
                &pair.sum_dim_intlist([1i64].as_slice(), false, Kind::Float),
                &pair.sum_dim_intlist([2i64].as_slice(), false, Kind::Float),
            ],
            -1,
        ) / count;
        wire += pair_to_wire.apply(&self.pair_to_wire) * valid.to_kind(Kind::Float).unsqueeze(-1);

        // One context token sums independent learned vectors keyed by feature and
        // reason. It needs no fictitious wire, pair, or group (even for empty groups).
        if let Some(projection) = &self.missing_projection {
            use crate::model::DeviceFeatureStatus::{Inapplicable, ResourceLimit, Unsupported};
            let indicators = Tensor::stack(
                &[
                    statuses.eq(Inapplicable as i64),
                    statuses.eq(Unsupported as i64),
                    statuses.eq(ResourceLimit as i64),
                ],
                2,
            )
            .flatten(1, 2);
            context_masks.push(indicators.any_dim(1, true));
            context_tokens.push(
                indicators
                    .to_kind(Kind::Float)
                    .apply(projection)
                    .unsqueeze(1),
            );
        }
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
            (wire, pair) = block.forward(&wire, &pair, &valid, share_wire_projections);
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
