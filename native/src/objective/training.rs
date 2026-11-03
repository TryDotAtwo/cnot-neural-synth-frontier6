//! Full-graph batch validation and differentiable loss construction.
mod preflight;
mod record_loss;

use super::{FullGraphObjective, ObjectiveDiagnostics, StateAdmission};
use crate::alphabet::CnotAlphabet;
use crate::features::FeatureRegistry;
use crate::model::FusionDistanceField;
use crate::nn_training::PreparedTrainingLoss;
use crate::state::MatrixState;
use crate::training_data::TrainingRecord;
use tch::nn::VarStore;
use tch::Tensor;

struct Validated<'a> {
    record: &'a TrainingRecord,
    alphabet: &'a CnotAlphabet,
    lower_bound: u64,
    upper_bound: u64,
    successors: Vec<MatrixState>,
}

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

        let validated = preflight::validate_batch(
            &self.alphabets,
            &self.exact_catalogs,
            &self.forbidden,
            records,
            admission,
        )?;

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
            let record_loss = record_loss::compute_record_loss(
                self,
                item,
                model,
                variables,
                registry,
                &mut diagnostics,
            )?;
            if record_loss.term.requires_grad() {
                gradient_states.extend(record_loss.gradient_states);
            }
            terms.push(record_loss.term);
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
