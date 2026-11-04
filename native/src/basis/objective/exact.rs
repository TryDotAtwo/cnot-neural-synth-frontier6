use super::super::{BasisChoiceKind, BasisSide, BasisTemplate};
use super::{BasisLossAudit, BasisLossBackend, BasisLossOptions, BasisLossResult};
use crate::model::ModelError;
use tch::{Device, Kind, Tensor};

pub(super) fn exact_messages_loss(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    theta: &Tensor,
    options: BasisLossOptions,
) -> Result<BasisLossResult, ModelError> {
    if options.max_evaluations < 1 {
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
        .and_then(|bytes| {
            template
                .source_wires
                .len()
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|column_bytes| bytes.checked_add(column_bytes))
        })
        .ok_or_else(|| ModelError("resource_limit: exact workspace estimate overflow".into()))?;
    if estimated_workspace_bytes > options.max_bytes {
        return Err(ModelError(format!(
            "resource_limit: exact message workspace {estimated_workspace_bytes} bytes exceeds {}",
            options.max_bytes
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
            let mut column_codes = vec![0usize; template.source_wires.len()];
            for (target_row, row) in block.iter().enumerate() {
                for (code, &bit) in column_codes.iter_mut().zip(row) {
                    *code |= usize::from(bit) << target_row;
                }
            }
            for code in column_codes {
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
        loss = std::ops::Add::add(loss, theta.select(0, index as i64) * price);
    }
    let value = loss
        .detach()
        .to_device(Device::Cpu)
        .to_kind(Kind::Double)
        .double_value(&[]);
    Ok(BasisLossResult {
        loss,
        audit: BasisLossAudit {
            binding: options.binding,
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
