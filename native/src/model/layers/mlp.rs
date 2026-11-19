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

    /// Same affine map as [pair(i,j), pair(j,i), wire(i), wire(j)].
    /// The two wire projections are shared across all incident pairs.
    pub(super) fn forward_shared_wires(&self, wire: &Tensor, pair: &Tensor) -> Tensor {
        let p = pair.size()[3];
        let d = wire.size()[2];
        let weight = &self.first.ws;
        let pairs = Tensor::cat(&[pair, &pair.transpose(1, 2)], -1);
        let mut hidden = pairs.linear(&weight.narrow(1, 0, 2 * p), self.first.bs.as_ref());
        let left = wire.linear(&weight.narrow(1, 2 * p, d), None::<&Tensor>);
        let right = wire.linear(&weight.narrow(1, 2 * p + d, d), None::<&Tensor>);
        hidden += left.unsqueeze(2);
        hidden += right.unsqueeze(1);
        hidden.gelu("none").apply(&self.second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device, Kind};

    #[test]
    fn shared_wire_projection_matches_direct_pairs_including_padding() {
        tch::manual_seed(327);
        let store = VarStore::new(Device::Cpu);
        let (batch, n, d, p) = (3, 5, 7, 3);
        let mlp = Mlp::new(&store.root(), 2 * (p + d), 17, p);
        let wire = Tensor::randn([batch, n, d], (Kind::Float, Device::Cpu));
        let pair = Tensor::randn([batch, n, n, p], (Kind::Float, Device::Cpu));
        let _ = wire.get(0).narrow(0, 3, 2).fill_(0.0);
        let left = wire.unsqueeze(2).expand([batch, n, n, d], true);
        let right = wire.unsqueeze(1).expand([batch, n, n, d], true);
        let reference = mlp.forward(&Tensor::cat(
            &[&pair, &pair.transpose(1, 2), &left, &right],
            -1,
        ));
        let result = mlp.forward_shared_wires(&wire, &pair);
        assert!(result.allclose(&reference, 1e-5, 1e-6, false));
    }

    fn assert_shared_wire_gradient_matches_direct(device: Device) {
        tch::manual_seed(7412);
        let store = VarStore::new(device);
        let (batch, n, d, p) = (3, 6, 9, 4);
        let mlp = Mlp::new(&store.root(), 2 * (p + d), 21, p);
        let initial_wire = Tensor::randn([batch, n, d], (Kind::Float, device));
        let initial_pair = Tensor::randn([batch, n, n, p], (Kind::Float, device));
        let _ = initial_wire.get(0).narrow(0, 4, 2).fill_(0.0);
        let _ = initial_pair.get(0).narrow(0, 4, 2).fill_(0.0);
        let weights = Tensor::randn([batch, n, n, p], (Kind::Float, device));

        let wire = initial_wire.detach().set_requires_grad(true);
        let pair = initial_pair.detach().set_requires_grad(true);
        let left = wire.unsqueeze(2).expand([batch, n, n, d], true);
        let right = wire.unsqueeze(1).expand([batch, n, n, d], true);
        let reference = mlp.forward(&Tensor::cat(
            &[&pair, &pair.transpose(1, 2), &left, &right], -1,
        ));
        (&reference * &weights).sum(Kind::Float).backward();
        let wire_grad = wire.grad().detach() * 1.0;
        let pair_grad = pair.grad().detach() * 1.0;
        let parameters = store.trainable_variables();
        let parameter_grads: Vec<_> = parameters.iter().map(|v| v.grad().detach() * 1.0).collect();
        for parameter in &parameters {
            let _ = parameter.grad().zero_();
        }

        let wire = initial_wire.detach().set_requires_grad(true);
        let pair = initial_pair.detach().set_requires_grad(true);
        let result = mlp.forward_shared_wires(&wire, &pair);
        (&result * &weights).sum(Kind::Float).backward();
        assert!(result.allclose(&reference, 2e-4, 2e-5, false), "output differs on {device:?}");
        assert!(wire.grad().allclose(&wire_grad, 2e-4, 2e-5, false), "wire gradient differs on {device:?}");
        assert!(pair.grad().allclose(&pair_grad, 2e-4, 2e-5, false), "pair gradient differs on {device:?}");
        for (parameter, expected) in parameters.iter().zip(&parameter_grads) {
            assert!(parameter.grad().allclose(expected, 2e-4, 5e-5, false),
                    "parameter gradient differs on {device:?}");
        }
    }

    #[test]
    fn shared_wire_projection_preserves_cpu_gradients() {
        assert_shared_wire_gradient_matches_direct(Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn shared_wire_projection_preserves_cuda_gradients() {
        crate::torch_device::require_cuda_device(0).unwrap();
        assert_shared_wire_gradient_matches_direct(Device::Cuda(0));
    }
}
