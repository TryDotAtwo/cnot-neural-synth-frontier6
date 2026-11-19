//! Full-graph batch validation and differentiable loss construction.
#[cfg(feature = "cuda")]
pub(super) mod device;
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
    /// Evaluation never trusts a checkpoint's compact counts as proof. Recheck
    /// the exact bound train panel and its complete primitive closure first.
    pub(crate) fn verify_prevalidated_device_closure(
        &mut self, records: &[TrainingRecord],
    ) -> Result<(), String> {
        if !self.config.prevalidated_device_closure {
            return Ok(());
        }
        self.closure_verified = false;
        if self.training_records_sha256.as_deref() != Some(super::training_records_digest(records).as_str()) {
            return Err("closure check requires the exact bound ordered training records".into());
        }
        for record in records {
            let alphabet = self.alphabets.get(record.state.alphabet_id())
                .ok_or("closure check encountered an unregistered alphabet")?;
            preflight::admitted_upper_bound(
                alphabet, self.exact_catalogs.get(alphabet.id()), &self.forbidden, record, None,
            )?;
        }
        self.closure_verified = true;
        Ok(())
    }

    pub(super) fn compute_loss(
        &mut self,
        model: &FusionDistanceField,
        variables: &VarStore,
        registry: &FeatureRegistry,
        records: &[TrainingRecord],
        admission: Option<&dyn StateAdmission>,
    ) -> Result<PreparedTrainingLoss, String> {
        if self.config.sequential_batch_gradients {
            return Err("sequential batch gradients are only implemented by the CUDA runner".into());
        }
        if self.config.sampled_neighbor_count.is_some() {
            return Err("sampled Bellman training is only implemented by the CUDA runner".into());
        }
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

#[cfg(test)]
mod closure_audit_tests {
    use super::*;
    use crate::{CnotGate, FeaturePolicy};
    use crate::alphabet::WordCertificate;
    use crate::training_data::TrainingSplit;

    fn fixture() -> (CnotAlphabet, TrainingRecord, MatrixState) {
        let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
        let gate = CnotGate { control: 0, target: 1 };
        let identity = MatrixState::identity(3, alphabet.id()).unwrap();
        let mut root = identity.clone();
        root.apply_cnot(gate).unwrap();
        let record = TrainingRecord {
            state: root.clone(), split: TrainingSplit::Train, lineage_id: "fixture".into(),
            exact_distance: None,
            verified_words: vec![WordCertificate {
                start: root, end: identity.clone(), gates: vec![gate], physical_cost: 1,
                producer: "fixture".into(), lineage_id: "fixture".into(),
            }],
        };
        (alphabet, record, identity)
    }

    fn objective(alphabet: CnotAlphabet, forbidden: Vec<MatrixState>) -> FullGraphObjective {
        let registry = FeatureRegistry::new();
        FullGraphObjective::new(
            vec![alphabet], &registry,
            FeaturePolicy::all_model_inputs(crate::schema::MISSING_TOKEN_LAYOUT_ID),
            vec![], forbidden,
            super::super::FullGraphObjectiveConfig {
                require_all_model_inputs: false,
                compact_exposures: true,
                prevalidated_device_closure: true,
                ..Default::default()
            },
        ).unwrap()
    }

    #[test]
    fn prevalidated_closure_is_pending_until_checked_and_after_restore() {
        let (alphabet, record, _) = fixture();
        let records = vec![record.clone()];
        let mut objective = objective(alphabet, vec![]);
        objective.bind_training_records(&records).unwrap();
        assert!(!objective.audit().strict_unseen);
        objective.verify_prevalidated_device_closure(&records).unwrap();
        let verified = objective.audit();
        assert!(verified.strict_unseen);
        assert_eq!(verified.audit_mode, "prevalidated_device_one_step_closure");
        assert_eq!(verified.set_sha256.get("closure_contract"), Some(&objective.config_sha256));
        let bytes = objective.checkpoint_state_bytes().unwrap();
        objective.restore_checkpoint_state(&bytes).unwrap();
        assert!(!objective.audit().strict_unseen);
        let mut changed = record;
        changed.lineage_id = "different".into();
        assert!(objective.verify_prevalidated_device_closure(&[changed]).is_err());
        assert!(!objective.audit().strict_unseen);
        objective.verify_prevalidated_device_closure(&records).unwrap();
        assert_eq!(objective.audit(), verified);
    }

    #[test]
    fn prevalidated_closure_rejects_forbidden_root_and_neighbor() {
        let (alphabet, record, neighbor) = fixture();
        for forbidden in [record.state.clone(), neighbor] {
            let mut objective = objective(alphabet.clone(), vec![forbidden]);
            objective.bind_training_records(std::slice::from_ref(&record)).unwrap();
            assert!(objective.verify_prevalidated_device_closure(std::slice::from_ref(&record)).is_err());
            assert!(!objective.audit().strict_unseen);
            assert!(objective.audit().set_sha256.is_empty());
        }
    }
}
