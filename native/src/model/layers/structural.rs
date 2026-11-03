//! Pair-aware graph block used by the structural stem and fusion stack.

use super::initialization::pytorch_linear_config;
use super::mlp::Mlp;
use crate::model::ModelConfig;
use tch::nn::{self, Module};
use tch::{Kind, Tensor};

#[derive(Debug)]
pub(crate) struct StructuralBlock {
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
    pub(in crate::model) fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
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

    pub(in crate::model) fn forward(
        &self,
        wire: &Tensor,
        pair: &Tensor,
        valid: &Tensor,
    ) -> (Tensor, Tensor) {
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
