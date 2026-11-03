//! Two-layer GELU projection used by the structural and family heads.

use super::initialization::pytorch_linear_config;
use tch::nn;
use tch::Tensor;

#[derive(Debug)]
pub(crate) struct Mlp {
    first: nn::Linear,
    second: nn::Linear,
}

impl Mlp {
    pub(in crate::model) fn new(path: &nn::Path<'_>, input: i64, hidden: i64, output: i64) -> Self {
        Self {
            first: nn::linear(path / "0", input, hidden, pytorch_linear_config(input)),
            second: nn::linear(path / "2", hidden, output, pytorch_linear_config(hidden)),
        }
    }

    pub(in crate::model) fn forward(&self, input: &Tensor) -> Tensor {
        input.apply(&self.first).gelu("none").apply(&self.second)
    }
}
