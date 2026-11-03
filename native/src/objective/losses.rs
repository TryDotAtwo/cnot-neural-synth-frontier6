use crate::alphabet::CnotAlphabet;
use crate::bellman::BellmanBranchAudit;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::FusionDistanceField;
use crate::state::{CnotGate, MatrixState};
use crate::training_data::TrainingRecord;
use std::collections::HashMap;
use tch::{Kind, Reduction, Tensor};

pub(super) fn qhat_loss(
    model: &FusionDistanceField,
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    state: &MatrixState,
    output: &crate::model::FieldOutput,
    branch: &BellmanBranchAudit,
) -> Result<Tensor, String> {
    let mut targets = Vec::new();
    if state.is_identity() {
        let mut successors = Vec::with_capacity(alphabet.gates().len());
        let mut costs = Vec::with_capacity(alphabet.gates().len());
        for (index, gate) in alphabet.gates().iter().copied().enumerate() {
            let mut successor = state.clone();
            successor
                .apply_cnot(gate)
                .map_err(|error| error.to_string())?;
            successors.push(successor);
            costs.push(f64::from(
                alphabet
                    .cost_at(index)
                    .ok_or_else(|| "alphabet gate cost is missing".to_owned())?,
            ));
        }
        if !successors.is_empty() {
            let packet = registry
                .build(&successors, policy)
                .map_err(|error| error.to_string())?;
            let values = tch::no_grad(|| model.forward(&packet))
                .map_err(|error| error.to_string())?
                .value;
            if values.numel() != successors.len() || values.isfinite().all().int64_value(&[]) == 0 {
                return Err("nonfinite terminal Q-hat target values".into());
            }
            for (index, gate) in alphabet.gates().iter().copied().enumerate() {
                targets.push((gate, costs[index] + values.double_value(&[index as i64])));
            }
        }
    } else {
        if branch.primitive_totals.len() != alphabet.gates().len() {
            return Err(
                "complete Bellman selection did not retain every primitive Q-hat target".into(),
            );
        }
        targets.extend(
            alphabet
                .gates()
                .iter()
                .copied()
                .zip(branch.primitive_totals.iter().copied()),
        );
    }
    masked_smooth_l1(&output.qhat, &output.action_mask, &targets)
}

fn masked_smooth_l1(
    predictions: &Tensor,
    action_mask: &Tensor,
    targets: &[(CnotGate, f64)],
) -> Result<Tensor, String> {
    if predictions.size() != action_mask.size() || predictions.size().first() != Some(&1) {
        return Err("Q-hat/action-mask tensor shape mismatch".into());
    }
    if targets.is_empty() {
        return Ok(predictions
            .masked_fill(&Tensor::ones_like(predictions).to_kind(Kind::Bool), 0.0)
            .sum(predictions.kind())
            * 0.0);
    }
    let nodes = usize::try_from(predictions.size()[1])
        .map_err(|_| "Q-hat node dimension exceeds host index range")?;
    let mut indices = Vec::with_capacity(targets.len());
    let mut values = Vec::with_capacity(targets.len());
    for (gate, target) in targets {
        if !target.is_finite() || gate.control as usize >= nodes || gate.target as usize >= nodes {
            return Err("invalid Q-hat target or gate index".into());
        }
        let index = usize::try_from(gate.control)
            .ok()
            .and_then(|control| control.checked_mul(nodes))
            .and_then(|offset| offset.checked_add(gate.target as usize))
            .ok_or_else(|| "Q-hat action index overflow".to_owned())?;
        indices.push(i64::try_from(index).map_err(|_| "Q-hat index exceeds tensor range")?);
        values.push(*target);
    }
    let indices = Tensor::from_slice(&indices).to_device(predictions.device());
    let allowed = action_mask
        .view([-1])
        .index_select(0, &indices)
        .all()
        .int64_value(&[]);
    if allowed == 0 {
        return Err("Q-hat target contains a gate masked by the model".into());
    }
    let targets = Tensor::from_slice(&values)
        .to_device(predictions.device())
        .to_kind(predictions.kind());
    let selected = predictions.view([-1]).index_select(0, &indices);
    Ok(selected.smooth_l1_loss(&targets, Reduction::Mean, 1.0))
}

pub(super) fn teacher_loss(
    logits: &Tensor,
    action_mask: &Tensor,
    record: &TrainingRecord,
    alphabet: &CnotAlphabet,
    temperature: f64,
) -> Result<(Tensor, bool), String> {
    let mut first_costs: HashMap<CnotGate, u64> = HashMap::new();
    for word in &record.verified_words {
        if let Some(gate) = word.gates.first().copied() {
            first_costs
                .entry(gate)
                .and_modify(|cost| *cost = (*cost).min(word.physical_cost))
                .or_insert(word.physical_cost);
        }
    }
    if first_costs.is_empty() {
        return Ok((
            logits
                .masked_fill(&action_mask.logical_not(), 0.0)
                .sum(logits.kind())
                * 0.0,
            false,
        ));
    }
    let mut gates: Vec<_> = first_costs.into_iter().collect();
    gates.sort_by_key(|(gate, _)| (gate.control, gate.target));
    let best = gates
        .iter()
        .map(|(_, cost)| *cost)
        .min()
        .ok_or_else(|| "teacher gate set unexpectedly empty".to_owned())?;
    let nodes = usize::try_from(logits.size()[1])
        .map_err(|_| "proposal node dimension exceeds host index range")?;
    let mut indices = Vec::with_capacity(gates.len());
    let mut scores = Vec::with_capacity(gates.len());
    for (gate, cost) in &gates {
        if gate.control as usize >= nodes
            || gate.target as usize >= nodes
            || alphabet.cost(*gate).is_none()
        {
            return Err("teacher word starts with an action outside the alphabet".into());
        }
        let index = usize::try_from(gate.control)
            .ok()
            .and_then(|control| control.checked_mul(nodes))
            .and_then(|offset| offset.checked_add(gate.target as usize))
            .ok_or_else(|| "teacher proposal action index overflow".to_owned())?;
        indices.push(i64::try_from(index).map_err(|_| "teacher index exceeds tensor range")?);
        scores.push(-((*cost).saturating_sub(best) as f64) / temperature);
    }
    let indices = Tensor::from_slice(&indices).to_device(logits.device());
    if action_mask
        .view([-1])
        .index_select(0, &indices)
        .all()
        .int64_value(&[])
        == 0
    {
        return Err("teacher action is outside the model's valid action mask".into());
    }
    let target = Tensor::from_slice(&scores)
        .to_device(logits.device())
        .to_kind(logits.kind())
        .softmax(0, logits.kind());
    let selected_log_probs = logits
        .view([-1])
        .index_select(0, &indices)
        .log_softmax(0, logits.kind());
    let loss = -(target * selected_log_probs).sum(logits.kind());
    Ok((loss, true))
}
