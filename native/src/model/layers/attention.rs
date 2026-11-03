//! Reference-layout multi-head attention with an optional pairwise logit bias.

use super::initialization::{pytorch_linear_config, xavier_uniform};
use tch::nn;
use tch::{Kind, Tensor};

#[derive(Debug)]
pub(crate) struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: nn::Linear,
    heads: i64,
    width: i64,
}

impl MultiHeadAttention {
    pub(in crate::model) fn new(path: &nn::Path<'_>, width: i64, heads: i64) -> Self {
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

    pub(in crate::model) fn forward(
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
