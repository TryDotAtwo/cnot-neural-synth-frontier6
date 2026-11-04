use super::super::{pytorch_linear_config, ModelConfig, MultiHeadAttention};
use tch::nn::{self, Module};
use tch::Tensor;

#[derive(Debug)]
pub(super) struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    attention_norm: nn::LayerNorm,
    first: nn::Linear,
    second: nn::Linear,
    feedforward_norm: nn::LayerNorm,
}

impl TransformerEncoderLayer {
    pub(super) fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
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

    pub(super) fn forward(&self, input: &Tensor, valid: &Tensor) -> Tensor {
        let attended = self.attention.forward(input, input, input, valid, None);
        let hidden = self.attention_norm.forward(&(input + attended));
        self.feedforward_norm
            .forward(&(&hidden + hidden.apply(&self.first).relu().apply(&self.second)))
    }
}
