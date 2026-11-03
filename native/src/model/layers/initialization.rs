//! Parameter initialization kept compatible with the reference torch modules.

use tch::nn;

pub(super) fn xavier_uniform(input: i64, output: i64) -> nn::Init {
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
