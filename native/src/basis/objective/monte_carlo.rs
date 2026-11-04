use super::super::{BasisChoiceKind, BasisSide, BasisTemplate};
use super::{BasisLossAudit, BasisLossBackend, BasisLossOptions, BasisLossResult};
use crate::model::ModelError;
use tch::Tensor;

pub(super) fn monte_carlo_loss(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    theta: &Tensor,
    probabilities: &[f64],
    options: BasisLossOptions,
) -> Result<BasisLossResult, ModelError> {
    let BasisLossOptions {
        binding,
        samples,
        seed,
        max_evaluations,
        max_bytes,
        ..
    } = options;
    if samples == 0 {
        return Err(ModelError(
            "positive Monte Carlo sample count required".into(),
        ));
    }
    let required = samples
        .checked_mul(template.choices.len().saturating_mul(2).saturating_add(1))
        .ok_or_else(|| {
            ModelError("resource_limit: Monte Carlo evaluation count overflow".into())
        })?;
    if required > max_evaluations {
        return Err(ModelError(format!(
            "resource_limit: complete Monte Carlo gradient needs {required} evaluations, budget is {max_evaluations}"
        )));
    }
    let n_squared = template
        .n
        .checked_mul(template.n)
        .ok_or_else(|| ModelError("resource_limit: matrix workspace estimate overflow".into()))?;
    let estimated_workspace_bytes = n_squared
        .checked_mul(64)
        .and_then(|bytes| {
            samples
                .checked_mul(template.choices.len().saturating_add(1))
                .and_then(|items| items.checked_mul(8))
                .and_then(|sample_bytes| bytes.checked_add(sample_bytes))
        })
        .ok_or_else(|| {
            ModelError("resource_limit: Monte Carlo workspace estimate overflow".into())
        })?;
    if estimated_workspace_bytes > max_bytes {
        return Err(ModelError(format!(
            "resource_limit: Monte Carlo workspace {estimated_workspace_bytes} bytes exceeds {max_bytes}"
        )));
    }

    let mut rng = SplitMix64::new(seed);
    let mut costs = Vec::with_capacity(samples);
    let mut gradients = vec![Vec::with_capacity(samples); template.choices.len()];
    for _ in 0..samples {
        let bits = probabilities
            .iter()
            .map(|probability| rng.next_f64() < *probability)
            .collect::<Vec<_>>();
        costs.push(physical_cost(template, block, &bits)? as f64);
        for index in 0..bits.len() {
            let mut zero = bits.clone();
            let mut one = bits.clone();
            zero[index] = false;
            one[index] = true;
            let low = physical_cost(template, block, &zero)?;
            let high = physical_cost(template, block, &one)?;
            gradients[index].push((high as f64) - (low as f64));
        }
    }

    let mean_cost = mean(&costs);
    let cost_se = standard_error(&costs);
    let gradient_means = gradients
        .iter()
        .map(|values| mean(values))
        .collect::<Vec<_>>();
    let gradient_se = (samples > 1).then(|| {
        gradients
            .iter()
            .map(|values| standard_error(values).unwrap_or(0.0))
            .collect::<Vec<_>>()
    });
    let gradient = Tensor::from_slice(&gradient_means)
        .to_kind(theta.kind())
        .to_device(theta.device());
    let estimate = Tensor::from(mean_cost)
        .to_kind(theta.kind())
        .to_device(theta.device());
    // This straight-through first-order surrogate has the source estimator's
    // unbiased gradient, but intentionally does not define an MC Hessian.
    let loss = estimate + ((theta - theta.detach()) * gradient).sum(theta.kind());

    Ok(BasisLossResult {
        loss,
        audit: BasisLossAudit {
            binding,
            backend: BasisLossBackend::MonteCarlo,
            physical_width: template.width(),
            samples,
            evaluations: required,
            estimated_workspace_bytes,
            expected_paid_cost_estimate: mean_cost,
            expected_cost_standard_error: cost_se,
            gradient_standard_error: gradient_se,
            gradient_kind:
                "unbiased independent-Bernoulli coordinate paired differences; first order only",
            rng: Some("SplitMix64-v1; distinct from NumPy default_rng/PCG64"),
        },
    })
}

fn physical_cost(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    bits: &[bool],
) -> Result<u64, ModelError> {
    if bits.len() != template.choices.len() {
        return Err(ModelError(
            "basis assignment width differs from template".into(),
        ));
    }
    let mut transformed = block.to_vec();
    let mut preparation_cost = 0u64;
    for (choice, selected) in template.choices.iter().zip(bits) {
        if !selected {
            continue;
        }
        preparation_cost = preparation_cost
            .checked_add(match choice.kind {
                BasisChoiceKind::Add => 1,
                BasisChoiceKind::Swap => 3,
            })
            .ok_or_else(|| ModelError("basis preparation cost overflow".into()))?;
        let control = choice.control;
        let target = choice.target;
        match choice.kind {
            BasisChoiceKind::Swap if template.side == BasisSide::Source => {
                for row in &mut transformed {
                    row.swap(control, target);
                }
            }
            BasisChoiceKind::Swap => transformed.swap(control, target),
            BasisChoiceKind::Add if template.side == BasisSide::Source => {
                for row in &mut transformed {
                    row[control] ^= row[target];
                }
            }
            BasisChoiceKind::Add => {
                let source = transformed[control].clone();
                for (value, bit) in transformed[target].iter_mut().zip(source) {
                    *value ^= bit;
                }
            }
        }
    }
    let core = transformed.iter().flatten().try_fold(0u64, |sum, bit| {
        sum.checked_add(u64::from(*bit))
            .ok_or_else(|| ModelError("basis core cost overflow".into()))
    })?;
    preparation_cost
        .checked_mul(2)
        .and_then(|prepare| prepare.checked_add(core))
        .ok_or_else(|| ModelError("basis physical cost overflow".into()))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_error(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let average = mean(values);
    let sample_variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Some((sample_variance / values.len() as f64).sqrt())
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}
