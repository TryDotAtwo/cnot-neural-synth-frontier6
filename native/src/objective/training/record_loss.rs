use super::super::losses::{qhat_loss, teacher_loss};
use super::super::{FullGraphObjective, ObjectiveDiagnostics};
use super::Validated;
use crate::bellman::{
    recompute_backup, select_backup_with_model_input_coverage, BellmanSelectionConfig,
};
use crate::features::FeatureRegistry;
use crate::model::FusionDistanceField;
use crate::state::MatrixState;
use tch::nn::VarStore;
use tch::{Kind, Reduction, Tensor};

pub(super) struct RecordLoss {
    pub(super) term: Tensor,
    pub(super) gradient_states: Vec<MatrixState>,
}

pub(super) fn compute_record_loss(
    objective: &FullGraphObjective,
    item: &Validated<'_>,
    model: &FusionDistanceField,
    variables: &VarStore,
    registry: &FeatureRegistry,
    diagnostics: &mut ObjectiveDiagnostics,
) -> Result<RecordLoss, String> {
    let state = &item.record.state;
    // Terminal Bellman rows have no selected successor, but terminal
    // Q-hat still targets every primitive action. Check their packets
    // before the loss pass, using bounded per-item memory.
    if objective.config.require_all_model_inputs
        && objective.config.qhat_weight > 0.0
        && state.is_identity()
    {
        let packet = registry
            .build(&item.successors, &objective.policy)
            .map_err(|error| error.to_string())?;
        packet
            .require_complete_model_input_coverage()
            .map_err(|error| error.to_string())?;
    }
    let selection = select_backup_with_model_input_coverage(
        model,
        variables,
        std::slice::from_ref(state),
        registry,
        &objective.policy,
        item.alphabet,
        BellmanSelectionConfig {
            macro_provider: None,
            chunk_size: Some(objective.config.neighbor_chunk_size),
            require_complete_model_input_coverage: objective.config.require_all_model_inputs,
        },
    )
    .map_err(|error| error.to_string())?;
    let branch = selection
        .branches()
        .next()
        .ok_or_else(|| "Bellman selection did not return a branch".to_owned())?;
    diagnostics.primitive_candidates = diagnostics
        .primitive_candidates
        .checked_add(branch.primitive_count)
        .ok_or_else(|| "primitive candidate counter overflow".to_owned())?;
    let bellman =
        recompute_backup(model, variables, &selection).map_err(|error| error.to_string())?;

    let parent_packet = selection
        .parent_packet(0)
        .ok_or_else(|| "Bellman selection did not retain its parent packet".to_owned())?;
    let output = model
        .forward(parent_packet)
        .map_err(|error| error.to_string())?;
    if output.value.numel() != 1 || output.value.isfinite().all().int64_value(&[]) == 0 {
        return Err("objective parent value is not a finite scalar".into());
    }
    let value = output.value.select(0, 0);
    let lower = Tensor::from_slice(&[item.lower_bound as f64])
        .to_device(value.device())
        .to_kind(value.kind())
        .select(0, 0);
    let upper = Tensor::from_slice(&[item.upper_bound as f64])
        .to_device(value.device())
        .to_kind(value.kind())
        .select(0, 0);
    let interval = (lower - &value).relu().pow_tensor_scalar(2.0)
        + (&value - upper).relu().pow_tensor_scalar(2.0);
    let exact_loss = if let Some(exact_distance) = item.record.exact_distance {
        diagnostics.exact_examples += 1;
        let target = Tensor::from_slice(&[exact_distance as f64])
            .to_device(value.device())
            .to_kind(value.kind())
            .select(0, 0);
        value.smooth_l1_loss(&target, Reduction::Mean, 1.0)
    } else {
        value.shallow_clone() * 0.0
    };

    let qhat_loss = if objective.config.qhat_weight > 0.0 {
        qhat_loss(
            model,
            registry,
            &objective.policy,
            item.alphabet,
            state,
            &output,
            branch,
        )?
    } else {
        output
            .qhat
            .masked_fill(&Tensor::ones_like(&output.qhat).to_kind(Kind::Bool), 0.0)
            .sum(output.qhat.kind())
            * 0.0
    };
    let teacher = if objective.config.teacher_weight > 0.0 {
        let (loss, used) = teacher_loss(
            &output.proposal_logits,
            &output.action_mask,
            item.record,
            item.alphabet,
            objective.config.teacher_temperature,
        )?;
        diagnostics.teacher_examples += usize::from(used);
        loss
    } else {
        output
            .proposal_logits
            .masked_fill(&output.action_mask.logical_not(), 0.0)
            .sum(output.proposal_logits.kind())
            * 0.0
    };

    let term = bellman.loss * objective.config.bellman_weight
        + (interval + exact_loss) * objective.config.supervision_weight
        + qhat_loss * objective.config.qhat_weight
        + teacher * objective.config.teacher_weight;

    let gradient_states = if term.requires_grad() {
        selection.gradient_exposure_states()
    } else {
        Vec::new()
    };
    Ok(RecordLoss {
        term,
        gradient_states,
    })
}
