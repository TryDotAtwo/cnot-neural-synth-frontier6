use crate::alphabet::CnotAlphabet;
use crate::state::CnotGate;
use crate::training_data::TrainingRecord;
use std::collections::HashMap;
use tch::Tensor;

/// Certificate-derived label metadata. Subtract integer costs before conversion
/// so adjacent large costs cannot collapse to the same teacher label.
pub(in crate::objective) fn first_gate_labels(
    record: &TrainingRecord,
    alphabet: &CnotAlphabet,
    nodes: usize,
) -> Result<(Vec<i64>, Vec<f64>), String> {
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
        return Ok((Vec::new(), Vec::new()));
    }
    let mut gates: Vec<_> = first_costs.into_iter().collect();
    gates.sort_by_key(|(gate, _)| (gate.control, gate.target));
    let best = gates
        .iter()
        .map(|(_, cost)| *cost)
        .min()
        .ok_or_else(|| "teacher gate set unexpectedly empty".to_owned())?;
    let mut indices = Vec::with_capacity(gates.len());
    let mut deltas = Vec::with_capacity(gates.len());
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
        deltas.push((*cost - best) as f64);
    }
    Ok((indices, deltas))
}

pub(crate) fn teacher_loss(
    logits: &Tensor,
    action_mask: &Tensor,
    record: &TrainingRecord,
    alphabet: &CnotAlphabet,
    temperature: f64,
) -> Result<(Tensor, bool), String> {
    let nodes = usize::try_from(logits.size()[1])
        .map_err(|_| "proposal node dimension exceeds host index range")?;
    let (indices, deltas) = first_gate_labels(record, alphabet, nodes)?;
    if indices.is_empty() {
        return Ok((
            logits
                .masked_fill(&action_mask.logical_not(), 0.0)
                .sum(logits.kind())
                * 0.0,
            false,
        ));
    }
    let scores: Vec<_> = deltas
        .into_iter()
        .map(|delta| -delta / temperature)
        .collect();
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
        .masked_fill(&action_mask.logical_not(), f64::NEG_INFINITY)
        .view([-1])
        .log_softmax(0, logits.kind())
        .index_select(0, &indices);
    let loss = -(target * selected_log_probs).sum(logits.kind());
    Ok((loss, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_data::TrainingSplit;
    use crate::{MatrixState, WordCertificate};
    use tch::{Device, Kind};

    #[test]
    fn one_verified_action_competes_with_all_legal_actions() {
        let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
        let goal = MatrixState::identity(2, alphabet.id()).unwrap();
        let gate = alphabet.gates()[0];
        let mut start = goal.clone();
        start.apply_cnot(gate).unwrap();
        let record = TrainingRecord {
            state: start.clone(),
            split: TrainingSplit::Train,
            verified_words: vec![WordCertificate {
                start,
                end: goal,
                gates: vec![gate],
                physical_cost: 1,
                producer: "test".into(),
                lineage_id: "teacher-one-action".into(),
            }],
            exact_distance: None,
            lineage_id: "teacher-one-action".into(),
        };
        let logits = Tensor::from_slice(&[99f32, 0., 0., 99.])
            .view([1, 2, 2])
            .to_device(Device::Cpu)
            .set_requires_grad(true);
        let mask = Tensor::from_slice(&[0i64, 1, 1, 0])
            .view([1, 2, 2])
            .to_kind(Kind::Bool);
        let (loss, used) = teacher_loss(&logits, &mask, &record, &alphabet, 1.0).unwrap();
        assert!(used);
        assert!((loss.double_value(&[]) - std::f64::consts::LN_2).abs() < 1e-6);
        loss.backward();
        let gradient = logits.grad().view([-1]);
        assert!((gradient.double_value(&[1]) + 0.5).abs() < 1e-6);
        assert!((gradient.double_value(&[2]) - 0.5).abs() < 1e-6);
        assert_eq!(gradient.double_value(&[0]), 0.0);
        assert_eq!(gradient.double_value(&[3]), 0.0);
    }
}
