use super::ModelConfig;
use tch::nn::{self, Module};
use tch::{Kind, Tensor};

fn xavier_uniform(input: i64, output: i64) -> nn::Init {
    let bound = (6.0 / (input + output) as f64).sqrt();
    nn::Init::Uniform {
        lo: -bound,
        up: bound,
    }
}

pub(crate) fn pytorch_linear_config(input: i64) -> nn::LinearConfig {
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
pub(crate) struct Mlp {
    first: nn::Linear,
    second: nn::Linear,
}

impl Mlp {
    pub(super) fn new(path: &nn::Path<'_>, input: i64, hidden: i64, output: i64) -> Self {
        Self {
            first: nn::linear(path / "0", input, hidden, pytorch_linear_config(input)),
            second: nn::linear(path / "2", hidden, output, pytorch_linear_config(hidden)),
        }
    }

    pub(super) fn forward(&self, input: &Tensor) -> Tensor {
        input.apply(&self.first).gelu("none").apply(&self.second)
    }
}

#[derive(Debug)]
pub(crate) struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: nn::Linear,
    heads: i64,
    width: i64,
}

impl MultiHeadAttention {
    pub(super) fn new(path: &nn::Path<'_>, width: i64, heads: i64) -> Self {
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

    pub(super) fn forward(
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
    pub(super) fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
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

    pub(super) fn forward(&self, wire: &Tensor, pair: &Tensor, valid: &Tensor) -> (Tensor, Tensor) {
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
