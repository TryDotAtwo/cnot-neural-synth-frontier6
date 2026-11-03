use super::{Mlp, ModelConfig, AUXILIARY_HIDDEN_DIM};
use tch::nn;
use tch::{Kind, Tensor};

#[derive(Debug)]
pub(super) struct AuxiliaryHeads {
    proposal: Mlp,
    qhat: Mlp,
    wire_dim: i64,
}

impl AuxiliaryHeads {
    pub(super) fn new(path: &nn::Path<'_>, config: &ModelConfig) -> Self {
        let input = 2 * config.wire_dim + 2 * config.pair_dim + 1;
        Self {
            proposal: Mlp::new(&(path / "proposal"), input, AUXILIARY_HIDDEN_DIM, 1),
            qhat: Mlp::new(&(path / "qhat"), input, AUXILIARY_HIDDEN_DIM, 1),
            wire_dim: config.wire_dim,
        }
    }

    pub(super) fn forward(
        &self,
        wire: &Tensor,
        pair: &Tensor,
        value: &Tensor,
        valid: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let shape = wire.size();
        let batch = shape[0];
        let nodes = shape[1];
        let control = wire
            .unsqueeze(2)
            .expand([batch, nodes, nodes, self.wire_dim], true);
        let target = wire
            .unsqueeze(1)
            .expand([batch, nodes, nodes, self.wire_dim], true);
        let shared_value = value
            .detach()
            .view([batch, 1, 1, 1])
            .expand([batch, nodes, nodes, 1], true);
        let input = Tensor::cat(
            &[
                &control,
                &target,
                pair,
                &pair.transpose(1, 2),
                &shared_value,
            ],
            -1,
        );
        let mut action_mask = valid.unsqueeze(2).logical_and(&valid.unsqueeze(1));
        let eye = Tensor::eye(nodes, (Kind::Bool, valid.device()));
        action_mask = action_mask.logical_and(&eye.logical_not().unsqueeze(0));
        let logits = self
            .proposal
            .forward(&input)
            .squeeze_dim(-1)
            .masked_fill(&action_mask.logical_not(), f64::NEG_INFINITY);
        let qhat = (self.qhat.forward(&input).squeeze_dim(-1).softplus() + 1.0)
            .masked_fill(&action_mask.logical_not(), f64::INFINITY);
        (logits, qhat, action_mask)
    }
}
