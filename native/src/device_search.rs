//! Bounded search in the complete, unit-cost CNOT graph. A strict shortlist uses
//! the learned action head, then V ranks full successor matrices. Exact closed
//! states and first-generated representatives live on the same CUDA device.
mod frontier;
mod runtime;
mod scoring;
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_state::DeviceStateBatch;
use crate::model::FusionDistanceField;
use crate::{CnotAlphabet, CnotGate, MatrixState, WordCertificate};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BeamConfig {
    pub beam_width: i64,
    pub shortlist: i64,
    pub max_depth: i64,
    pub max_expanded: i64,
    pub inference_batch: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeamStatus {
    Solved,
    DepthLimit,
    ResourceLimit,
    Exhausted,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionHead {
    Qhat,
    Proposal,
}

#[derive(Debug, Serialize)]
pub struct BeamDepth {
    pub depth: i64,
    pub frontier: i64,
    pub expanded: i64,
    pub generated: i64,
    pub value_evaluations: i64,
    pub retained: i64,
}

pub struct BeamResult {
    pub status: BeamStatus,
    /// Verified reduction in physical control/target orientation.
    pub word: Option<Vec<CnotGate>>,
    pub expanded: i64,
    pub generated: i64,
    pub value_evaluations: i64,
    pub action_evaluations: i64,
    /// Physical scoring slots, including padding in the bounded CUDA schedule.
    pub value_slots: i64,
    pub action_slots: i64,
    pub depths: Vec<BeamDepth>,
}

fn pack(state: &MatrixState, device: Device) -> Tensor {
    let data: Vec<_> = state.packed_rows().iter().map(|w| *w as i64).collect();
    Tensor::from_slice(&data)
        .view([
            1,
            i64::from(state.dimension()),
            state.words_per_row() as i64,
        ])
        .to_device(device)
}

/// Bound all score calls to the loaded complete model and its exact feature plan.
pub fn search_neural(
    root: &MatrixState,
    model: &FusionDistanceField,
    plan: &DeviceFeaturePlan,
    config: BeamConfig,
    head: ActionHead,
) -> Result<BeamResult, String> {
    let Device::Cuda(index) = model.device() else {
        return Err("learned search requires CUDA".into());
    };
    let alphabet = CnotAlphabet::all_to_all_unit(root.dimension()).map_err(|e| e.to_string())?;
    let n = i64::from(root.dimension());
    let positions: Vec<_> = alphabet
        .gates()
        .iter()
        .map(|g| i64::from(g.control) * n + i64::from(g.target))
        .collect();
    let positions = Tensor::from_slice(&positions).to_device(model.device());
    let field = |rows: &Tensor| {
        let batch = DeviceStateBatch::from_packed_tensor_for_alphabet(
            rows.shallow_clone(),
            &alphabet,
            index,
        )?;
        let packet = plan.build(&batch)?;
        let coverage = packet.first_coverage_failure().map_err(|e| e.to_string())?;
        let output = model.forward_device(&packet).map_err(|e| e.to_string())?;
        Ok::<_, String>((output, coverage))
    };
    search_with_coverage(
        root,
        index,
        config,
        plan.search_packet_capacity(root.dimension())?,
        |rows| {
            let (out, coverage) = field(rows)?;
            let action = match head {
                ActionHead::Qhat => out.qhat,
                ActionHead::Proposal => -out.proposal_logits,
            };
            Ok((action.flatten(1, -1).index_select(1, &positions), coverage))
        },
        |rows| {
            let (out, coverage) = field(rows)?;
            Ok((out.value.view([-1]), coverage))
        },
    )
}

/// Callbacks consume fixed-shape batches of packed rows. They must score each
/// row independently. The core never receives reference words or distances.
pub fn search_with_scorers(
    root: &MatrixState,
    index: usize,
    config: BeamConfig,
    mut actions: impl FnMut(&Tensor) -> Result<Tensor, String>,
    mut values: impl FnMut(&Tensor) -> Result<Tensor, String>,
) -> Result<BeamResult, String> {
    let admitted = |rows: &Tensor| Tensor::zeros([rows.size()[0]], (Kind::Int64, rows.device()));
    search_with_coverage(
        root,
        index,
        config,
        i64::MAX,
        |rows| Ok((actions(rows)?, admitted(rows))),
        |rows| Ok((values(rows)?, admitted(rows))),
    )
}

fn search_with_coverage(
    root: &MatrixState,
    index: usize,
    config: BeamConfig,
    max_packet_rows: i64,
    mut actions: impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
    mut values: impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
) -> Result<BeamResult, String> {
    if root.dimension() < 2
        || config.beam_width < 1
        || config.shortlist < 1
        || config.max_depth < 0
        || config.max_expanded < 1
        || config.inference_batch < 1
    {
        return Err("invalid CUDA beam extents or work budget".into());
    }
    let alphabet = CnotAlphabet::all_to_all_unit(root.dimension()).map_err(|e| e.to_string())?;
    if root.alphabet_id() != alphabet.id() {
        return Err("learned beam requires the bound all-to-all unit CNOT alphabet".into());
    }
    let identity =
        MatrixState::identity(root.dimension(), alphabet.id()).map_err(|e| e.to_string())?;
    crate::torch_device::require_cuda_device(index)?;
    let d = Device::Cuda(index);
    let a = alphabet.gates().len() as i64;
    let schedule =
        runtime::Schedule::new(config, a, root.packed_rows().len() as i64, max_packet_rows)?;
    let (host, rejected_state) = tch::no_grad(|| {
        runtime::run(
            &pack(root, d),
            &pack(&identity, d),
            config,
            &schedule,
            &mut actions,
            &mut values,
        )
    })?;
    let [code, expanded, generated, value_evaluations, action_evaluations, detail, failed_depth, ..] =
        host.as_slice()
    else {
        return Err("incomplete beam receipt".into());
    };
    let status = match *code {
        1 => BeamStatus::Solved,
        2 => BeamStatus::Exhausted,
        3 => BeamStatus::ResourceLimit,
        4 => BeamStatus::DepthLimit,
        5 | 6 => {
            let stage = if *code == 5 { "action" } else { "value" };
            let why = if *detail == 0 {
                "nonfinite model score".into()
            } else {
                let specs = crate::schema::all_model_input_specs();
                let id = specs
                    .get((detail / 256 - 1) as usize)
                    .ok_or("invalid coverage slot in beam receipt")?
                    .id;
                let status = match detail % 256 {
                    0 => "not_requested",
                    3 => "unsupported",
                    4 => "resource_limit",
                    _ => "invalid_status",
                };
                format!("{id} {status}")
            };
            let signed = Vec::<i64>::try_from(rejected_state.view([-1]).to_device(Device::Cpu))
                .map_err(|e| e.to_string())?;
            let packed: Vec<_> = signed.into_iter().map(|w| w as u64).collect();
            return Err(format!(
                "depth {failed_depth} {stage} scoring rejected packed rows {packed:?}: {why}"
            ));
        }
        _ => return Err("invalid beam status".into()),
    };
    let end = 7 + schedule.layers as usize;
    let word = if status == BeamStatus::Solved {
        let ids = &host[7..end];
        let len = ids.iter().position(|&id| id < 0).unwrap_or(ids.len());
        if ids[len..].iter().any(|&id| id != -1) || ids[..len].iter().any(|&id| id >= a) {
            return Err("invalid solved beam history".into());
        }
        let gates = ids[..len]
            .iter()
            .map(|&id| alphabet.gates()[id as usize])
            .collect();
        let proof = WordCertificate {
            start: root.clone(),
            end: identity,
            gates,
            physical_cost: len as u64,
            producer: "cuda_primitive_neural_beam_v1".into(),
            lineage_id: "post-search-replay".into(),
        };
        proof.verify(&alphabet).map_err(|e| e.to_string())?;
        Some(proof.gates)
    } else {
        None
    };
    let depths = host[end..]
        .chunks_exact(6)
        .enumerate()
        .filter(|(_, row)| row[0] > 0)
        .map(|(i, row)| BeamDepth {
            depth: i as i64 + 1,
            frontier: row[0],
            expanded: row[1],
            generated: row[2],
            value_evaluations: row[3],
            retained: row[4],
        })
        .collect();
    Ok(BeamResult {
        status,
        word,
        expanded: *expanded,
        generated: *generated,
        value_evaluations: *value_evaluations,
        action_evaluations: *action_evaluations,
        value_slots: schedule.value_slots,
        action_slots: schedule.action_slots,
        depths,
    })
}
