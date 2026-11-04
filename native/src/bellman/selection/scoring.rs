use super::super::BellmanError;
use super::candidates::CandidateSet;
use super::{candidate_total, BellmanSelectionConfig};
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::FusionDistanceField;
use crate::packet::FeaturePacket;
use tch::{Device, Kind};

pub(super) struct ScoredCandidates {
    pub(super) selected_index: Option<usize>,
    pub(super) selected_cost: Option<u64>,
    pub(super) selected_packet: Option<FeaturePacket>,
    pub(super) totals: Vec<f64>,
}

pub(super) fn score_candidates(
    model: &FusionDistanceField,
    candidates: &CandidateSet,
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    config: BellmanSelectionConfig,
) -> Result<ScoredCandidates, BellmanError> {
    let width = config.chunk_size.unwrap_or(candidates.successors.len());
    let mut best = f64::INFINITY;
    let mut selected_index = None;
    let mut selected_cost = None;
    let mut selected_packet = None;
    let mut totals = Vec::with_capacity(candidates.successors.len());
    for start in (0..candidates.successors.len()).step_by(width) {
        let end = (start + width).min(candidates.successors.len());
        let chunk = &candidates.successors[start..end];
        let packet = registry.build(chunk, policy)?;
        if config.require_complete_model_input_coverage {
            packet
                .require_complete_model_input_coverage()
                .map_err(|error| BellmanError::Invalid(error.to_string()))?;
        }
        let chunk_totals = tch::no_grad(|| -> Result<Vec<f64>, BellmanError> {
            let values = model.forward(&packet)?.value;
            if values.isfinite().all().int64_value(&[]) == 0 {
                return Err(BellmanError::Invalid(
                    "nonfinite Bellman candidate value".into(),
                ));
            }
            let flat = values
                .to_device(Device::Cpu)
                .to_kind(Kind::Double)
                .contiguous()
                .view([-1]);
            let values = Vec::<f64>::try_from(&flat)?;
            if values.len() != end - start {
                return Err(BellmanError::Invalid(
                    "model candidate count differs from successor chunk".into(),
                ));
            }
            values
                .into_iter()
                .zip(candidates.costs[start..end].iter().copied())
                .map(|(value, cost)| candidate_total(value, cost))
                .collect()
        })?;
        if chunk_totals.len() != end - start {
            return Err(BellmanError::Invalid(
                "model candidate count differs from successor chunk".into(),
            ));
        }
        for (local_index, total) in chunk_totals.iter().copied().enumerate() {
            totals.push(total);
            if total < best {
                best = total;
                let global_index = start + local_index;
                selected_index = Some(global_index);
                selected_cost = Some(candidates.costs[global_index]);
                selected_packet = Some(packet.single_row(local_index)?);
            }
        }
    }
    Ok(ScoredCandidates {
        selected_index,
        selected_cost,
        selected_packet,
        totals,
    })
}
