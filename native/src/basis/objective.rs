//! Differentiable expected paid cost for the independent Bernoulli basis head.
//!
//! The exact backend contracts the same one-sided messages as the source
//! objective. The sampled backend uses coordinate-paired cost differences and
//! exposes its RNG algorithm in the audit record; it is reproducible within
//! this implementation but does not claim NumPy bit-stream compatibility.

mod exact;
mod monte_carlo;
mod validation;

use super::{BasisBinding, BasisTemplate};
use crate::model::ModelError;
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisLossBackend {
    ExactMessages,
    MonteCarlo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisLossAudit {
    pub binding: BasisBinding,
    pub backend: BasisLossBackend,
    pub physical_width: usize,
    pub samples: usize,
    pub evaluations: usize,
    pub estimated_workspace_bytes: usize,
    pub expected_paid_cost_estimate: f64,
    pub expected_cost_standard_error: Option<f64>,
    pub gradient_standard_error: Option<Vec<f64>>,
    pub gradient_kind: &'static str,
    pub rng: Option<&'static str>,
}

pub(crate) struct BasisLossResult {
    pub loss: Tensor,
    pub audit: BasisLossAudit,
}

pub(crate) fn paid_cost_loss(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    probabilities: &Tensor,
    binding: BasisBinding,
    backend: BasisLossBackend,
    samples: usize,
    seed: u64,
    max_evaluations: usize,
    max_bytes: usize,
) -> Result<BasisLossResult, ModelError> {
    validation::validate_block(template, block)?;
    let theta = probabilities
        .detach()
        .to_device(Device::Cpu)
        .to_kind(Kind::Double);
    if theta.size() != [template.choices.len() as i64] {
        return Err(ModelError(
            "basis probability vector does not match ordered template choices".into(),
        ));
    }
    let theta_values = Vec::<f64>::try_from(&theta)
        .map_err(|error| ModelError(format!("cannot read basis probabilities: {error}")))?;
    if theta_values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(ModelError(
            "basis probabilities must be finite and lie in [0,1]".into(),
        ));
    }

    match backend {
        BasisLossBackend::ExactMessages => exact::exact_messages_loss(
            template,
            block,
            probabilities,
            binding,
            max_evaluations,
            max_bytes,
        ),
        BasisLossBackend::MonteCarlo => monte_carlo::monte_carlo_loss(
            template,
            block,
            probabilities,
            &theta_values,
            binding,
            samples,
            seed,
            max_evaluations,
            max_bytes,
        ),
    }
}
