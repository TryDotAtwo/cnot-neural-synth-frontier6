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

#[derive(Clone, Copy)]
pub(crate) struct BasisLossOptions {
    pub(crate) binding: BasisBinding,
    pub(crate) backend: BasisLossBackend,
    pub(crate) samples: usize,
    pub(crate) seed: u64,
    pub(crate) max_evaluations: usize,
    pub(crate) max_bytes: usize,
}

pub(crate) fn paid_cost_loss(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    probabilities: &Tensor,
    options: BasisLossOptions,
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

    match options.backend {
        BasisLossBackend::ExactMessages => {
            exact::exact_messages_loss(template, block, probabilities, options)
        }
        BasisLossBackend::MonteCarlo => {
            monte_carlo::monte_carlo_loss(template, block, probabilities, &theta_values, options)
        }
    }
}
