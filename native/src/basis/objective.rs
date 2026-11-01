//! Differentiable expected paid cost for the independent Bernoulli basis head.
//!
//! The exact backend contracts the same one-sided messages as the source
//! objective. The sampled backend uses coordinate-paired cost differences and
//! exposes its RNG algorithm in the audit record; it is reproducible within
//! this implementation but does not claim NumPy bit-stream compatibility.

use super::{BasisBinding, BasisChoiceKind, BasisSide, BasisTemplate};
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
    validate_block(template, block)?;
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
        BasisLossBackend::ExactMessages => exact_messages_loss(
            template,
            block,
            probabilities,
            binding,
            max_evaluations,
            max_bytes,
        ),
        BasisLossBackend::MonteCarlo => monte_carlo_loss(
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

fn validate_block(template: &BasisTemplate, block: &[Vec<u8>]) -> Result<(), ModelError> {
    if block.len() != template.target_wires.len()
        || block
            .iter()
            .any(|row| row.len() != template.source_wires.len() || row.iter().any(|bit| *bit > 1))
    {
        return Err(ModelError(
            "basis block must be a binary target-by-source matrix".into(),
        ));
    }
    Ok(())
}

fn exact_messages_loss(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    theta: &Tensor,
    binding: BasisBinding,
    max_evaluations: usize,
    max_bytes: usize,
) -> Result<BasisLossResult, ModelError> {
    if max_evaluations < 1 {
        return Err(ModelError(
            "resource_limit: exact contraction evaluation budget exhausted".into(),
        ));
    }
    let width = template.width();
    if width >= 63 {
        return Err(ModelError(
            "resource_limit: exact message index space unsupported; use Monte Carlo backend".into(),
        ));
    }
    let state_count = 1usize
        .checked_shl(width as u32)
        .ok_or_else(|| ModelError("resource_limit: exact message index space overflow".into()))?;
    let choice_count = template.choices.len();
    // Includes the retained choice permutations, forward states, and working
    // vectors; this deliberately bounds the allocations that the source
    // estimate omits for large choice counts.
    let estimated_workspace_bytes = state_count
        .checked_mul(choice_count.saturating_mul(2).saturating_add(5))
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| ModelError("resource_limit: exact workspace estimate overflow".into()))?;
    if estimated_workspace_bytes > max_bytes {
        return Err(ModelError(format!(
            "resource_limit: exact message workspace {estimated_workspace_bytes} bytes exceeds {max_bytes}"
        )));
    }

    let mut initial = vec![0.0f64; state_count];
    match template.side {
        BasisSide::Source => {
            for row in block {
                let mut code = 0usize;
                for (bit_index, bit) in row.iter().copied().enumerate() {
                    code |= usize::from(bit) << bit_index;
                }
                initial[code] += 1.0;
            }
        }
        BasisSide::Target => {
            for source_column in 0..template.source_wires.len() {
                let mut code = 0usize;
                for target_row in 0..template.target_wires.len() {
                    code |= usize::from(block[target_row][source_column]) << target_row;
                }
                initial[code] += 1.0;
            }
        }
    }

    let population = (0..state_count)
        .map(|value| f64::from(value.count_ones()))
        .collect::<Vec<_>>();
    let mut permutations = Vec::with_capacity(choice_count);
    for choice in &template.choices {
        let mut permutation = Vec::with_capacity(state_count);
        for value in 0..state_count {
            let value = value as u64;
            let control = choice.control as u32;
            let target = choice.target as u32;
            let mapped = match choice.kind {
                BasisChoiceKind::Swap => {
                    let differing = ((value >> control) ^ (value >> target)) & 1;
                    value ^ (differing << control) ^ (differing << target)
                }
                BasisChoiceKind::Add if template.side == BasisSide::Source => {
                    value ^ (((value >> target) & 1) << control)
                }
                BasisChoiceKind::Add => value ^ (((value >> control) & 1) << target),
            };
            permutation.push(i64::try_from(mapped).map_err(|_| {
                ModelError("resource_limit: exact message index exceeds tensor range".into())
            })?);
        }
        permutations.push(permutation);
    }

    let population = Tensor::from_slice(&population)
        .to_kind(theta.kind())
        .to_device(theta.device());
    let mut distribution = Tensor::from_slice(&initial)
        .to_kind(theta.kind())
        .to_device(theta.device());
    for (index, permutation) in permutations.iter().enumerate() {
        let indices = Tensor::from_slice(permutation).to_device(theta.device());
        let permuted = distribution.index_select(0, &indices);
        let probability = theta.select(0, index as i64);
        distribution =
            (probability.ones_like() - &probability) * &distribution + &probability * permuted;
    }

    let mut loss = (&distribution * population).sum(theta.kind());
    for (index, choice) in template.choices.iter().enumerate() {
        let price = match choice.kind {
            BasisChoiceKind::Add => 2.0,
            BasisChoiceKind::Swap => 6.0,
        };
        loss = loss + theta.select(0, index as i64) * price;
    }
    let value = loss
        .detach()
        .to_device(Device::Cpu)
        .to_kind(Kind::Double)
        .double_value(&[]);
    Ok(BasisLossResult {
        loss,
        audit: BasisLossAudit {
            binding,
            backend: BasisLossBackend::ExactMessages,
            physical_width: width,
            samples: 0,
            evaluations: 1,
            estimated_workspace_bytes,
            expected_paid_cost_estimate: value,
            expected_cost_standard_error: None,
            gradient_standard_error: None,
            gradient_kind: "exact independent-Bernoulli message contraction",
            rng: None,
        },
    })
}

fn monte_carlo_loss(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    theta: &Tensor,
    probabilities: &[f64],
    binding: BasisBinding,
    samples: usize,
    seed: u64,
    max_evaluations: usize,
    max_bytes: usize,
) -> Result<BasisLossResult, ModelError> {
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
