use super::{EncodedState, Mlp, ModelConfig, MultiHeadAttention, StructuralBlock};
use tch::nn::{self, Module};
use tch::{Kind, Tensor};

mod layer;
use layer::TransformerEncoderLayer;

#[derive(Debug)]
pub(super) struct FusionTransformer {
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
    pub(super) fn new(path: &nn::Path<'_>, config: &ModelConfig, family_count: usize) -> Self {
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

    pub(super) fn forward(
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
