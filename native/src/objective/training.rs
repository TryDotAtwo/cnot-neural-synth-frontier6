//! Full-graph batch validation and differentiable loss construction.
use super::losses::{qhat_loss, teacher_loss};
use super::{FullGraphObjective, ObjectiveDiagnostics, StateAdmission};
use crate::alphabet::CnotAlphabet;
use crate::bellman::{
    BellmanSelectionConfig, recompute_backup, select_backup_with_model_input_coverage,
};
use crate::features::FeatureRegistry;
use crate::model::FusionDistanceField;
use crate::nn_training::PreparedTrainingLoss;
use crate::packet::StateIdentity;
use crate::state::MatrixState;
use crate::training_data::{TrainingRecord, elementary_lower_bound};
use tch::nn::VarStore;
use tch::{Kind, Reduction, Tensor};

impl FullGraphObjective {
    pub(super) fn compute_loss(
        &mut self,
        model: &FusionDistanceField,
        variables: &VarStore,
        registry: &FeatureRegistry,
        records: &[TrainingRecord],
        admission: Option<&dyn StateAdmission>,
    ) -> Result<PreparedTrainingLoss, String> {
        if records.is_empty() {
            return Err("empty training batch".into());
        }
        if self.config.compact_exposures && admission.is_none() {
            return Err("compact exposure audit requires external admission".into());
        }
        if model.device() != variables.device() {
            return Err("model and variable store are on different devices".into());
        }
        if self.config.external_admission != admission.is_some() {
            return Err("runner external admission differs from the objective contract".into());
        }
        if admission.map(StateAdmission::contract_sha256)
            != self.config.external_admission_sha256.as_deref()
        {
            return Err("runner admission fingerprint differs from the objective contract".into());
        }

        struct Validated<'a> {
            record: &'a TrainingRecord,
            alphabet: &'a CnotAlphabet,
            lower_bound: u64,
            upper_bound: u64,
            successors: Vec<MatrixState>,
        }

        // Finish all split, certificate, bound, forbidden-neighbor, and
        // external-admission checks before mutating exposures or calling the model.
        let mut validated = Vec::with_capacity(records.len());
        for record in records {
            let alphabet = self
                .alphabets
                .get(record.state.alphabet_id())
                .ok_or_else(|| "training record uses an unregistered alphabet".to_owned())?;
            let exact = self.exact_catalogs.get(alphabet.id());
            let upper_bound = record
                .validate_for_training(alphabet, exact)
                .map_err(|error| error.to_string())?;
            let lower_bound =
                elementary_lower_bound(&record.state).map_err(|error| error.to_string())?;
            if lower_bound > upper_bound {
                return Err("verified word upper bound is below the source lower bound".into());
            }

            let mut successors = Vec::with_capacity(alphabet.gates().len());
            for gate in alphabet.gates().iter().copied() {
                let mut successor = record.state.clone();
                successor
                    .apply_cnot(gate)
                    .map_err(|error| error.to_string())?;
                if self.forbidden.contains(&StateIdentity::from(&successor)) {
                    return Err("forbidden state in complete backup; reject whole parent".into());
                }
                if let Some(admission) = admission {
                    admission.admit(&successor)?;
                }
                successors.push(successor);
            }
            if self.forbidden.contains(&StateIdentity::from(&record.state)) {
                return Err("forbidden state in training batch".into());
            }
            if let Some(admission) = admission {
                admission.admit(&record.state)?;
            }
            validated.push(Validated {
                record,
                alphabet,
                lower_bound,
                upper_bound,
                successors,
            });
        }

        // Feature packets are built once per example and retained only through
        // that example's loss construction. This keeps candidate coverage
        // fail-closed without multiplying provider work or holding every
        // Bellman packet in the batch at once.
        let mut terms = Vec::with_capacity(validated.len());
        let mut gradient_states = Vec::with_capacity(validated.len() * 2);
        let mut diagnostics = ObjectiveDiagnostics {
            examples: validated.len(),
            ..ObjectiveDiagnostics::default()
        };
        for item in &validated {
            let state = &item.record.state;
            // Terminal Bellman rows have no selected successor, but terminal
            // Q-hat still targets every primitive action. Check their packets
            // before the loss pass, using bounded per-item memory.
            if self.config.require_all_model_inputs
                && self.config.qhat_weight > 0.0
                && state.is_identity()
            {
                let packet = registry
                    .build(&item.successors, &self.policy)
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
                &self.policy,
                item.alphabet,
                BellmanSelectionConfig {
                    chunk_size: Some(self.config.neighbor_chunk_size),
                    require_complete_model_input_coverage: self.config.require_all_model_inputs,
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
            let bellman = recompute_backup(model, variables, &selection)
                .map_err(|error| error.to_string())?;

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

            let qhat_loss = if self.config.qhat_weight > 0.0 {
                qhat_loss(
                    model,
                    registry,
                    &self.policy,
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
            let teacher = if self.config.teacher_weight > 0.0 {
                let (loss, used) = teacher_loss(
                    &output.proposal_logits,
                    &output.action_mask,
                    item.record,
                    item.alphabet,
                    self.config.teacher_temperature,
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

            let term = bellman.loss * self.config.bellman_weight
                + (interval + exact_loss) * self.config.supervision_weight
                + qhat_loss * self.config.qhat_weight
                + teacher * self.config.teacher_weight;
            if term.requires_grad() {
                gradient_states.extend(selection.gradient_exposure_states());
            }
            terms.push(term);
        }
        if terms.is_empty() {
            return Err("empty full graph objective".into());
        }
        let loss_kind = terms[0].kind();
        let loss = Tensor::stack(&terms, 0).mean(loss_kind);
        if loss.numel() != 1 || loss.isfinite().all().int64_value(&[]) == 0 {
            return Err("nonfinite full graph objective loss".into());
        }
        // Commit exposure changes only after the complete batch has produced a
        // finite loss. A provider/model failure cannot leave a partial audit.
        for item in &validated {
            if self.config.supervision_weight > 0.0 {
                self.exposures
                    .observe(self.config.compact_exposures, 0, &item.record.state);
            }
            self.exposures
                .observe(self.config.compact_exposures, 1, &item.record.state);
            if !item.record.state.is_identity() || self.config.qhat_weight > 0.0 {
                for successor in &item.successors {
                    self.exposures
                        .observe(self.config.compact_exposures, 1, successor);
                }
            }
        }
        self.last_diagnostics = diagnostics;
        Ok(PreparedTrainingLoss {
            loss,
            gradient_states,
        })
    }
}
