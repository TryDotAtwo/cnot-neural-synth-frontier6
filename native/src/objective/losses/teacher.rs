use crate::alphabet::CnotAlphabet;
use crate::state::CnotGate;
use crate::training_data::TrainingRecord;
use std::collections::HashMap;
use tch::Tensor;

pub(crate) fn teacher_loss(
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
