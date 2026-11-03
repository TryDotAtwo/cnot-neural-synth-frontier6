use super::{FullGraphObjective, StateAdmission};
use crate::checkpoint::{CheckpointManifest, CheckpointStore};
use crate::features::FeatureRegistry;
use crate::model::FusionDistanceField;
use crate::nn_training::{
    DifferentiableObjective, NeuralTrainer, PreparedTrainingLoss, TchTrainingError,
};
use crate::state::MatrixState;
use crate::training_data::TrainingRecord;
use tch::nn::VarStore;

pub struct FullGraphObjectiveRunner<'a> {
    objective: &'a mut FullGraphObjective,
    model: &'a FusionDistanceField,
    variables: &'a VarStore,
    registry: &'a FeatureRegistry,
    records: &'a [TrainingRecord],
    admission: Option<&'a dyn StateAdmission>,
}

impl<'a> FullGraphObjectiveRunner<'a> {
    pub fn new(
        objective: &'a mut FullGraphObjective,
        model: &'a FusionDistanceField,
        variables: &'a VarStore,
        registry: &'a FeatureRegistry,
        records: &'a [TrainingRecord],
        admission: Option<&'a dyn StateAdmission>,
    ) -> Result<Self, String> {
        if records.is_empty() {
            return Err("training record set is empty".into());
        }
        if objective.config.compact_exposures && admission.is_none() {
            return Err("compact exposure audit requires external admission".into());
        }
        if objective.config.external_admission != admission.is_some() {
            return Err("runner external admission differs from the objective contract".into());
        }
        if admission.map(StateAdmission::contract_sha256)
            != objective.config.external_admission_sha256.as_deref()
        {
            return Err("runner admission fingerprint differs from the objective contract".into());
        }
        if objective.feature_contract_sha256 != registry.contract_sha256() {
            return Err("runner feature registry differs from the objective contract".into());
        }
        registry
            .selected_specs(&objective.policy)
            .map_err(|error| format!("objective feature policy is not executable: {error}"))?;
        if model.device() != variables.device() {
            return Err("model and variable store are on different devices".into());
        }
        objective.bind_training_records(records)?;
        Ok(Self {
            objective,
            model,
            variables,
            registry,
            records,
            admission,
        })
    }
}

impl DifferentiableObjective for FullGraphObjectiveRunner<'_> {
    fn compute(&mut self, indices: &[usize]) -> Result<PreparedTrainingLoss, String> {
        if indices.is_empty() {
            return Err("training sampler returned an empty batch".into());
        }
        let records = indices
            .iter()
            .map(|index| {
                self.records
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| "training sampler returned an out-of-range record".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.objective.compute_loss(
            self.model,
            self.variables,
            self.registry,
            &records,
            self.admission,
        )
    }

    fn observe_gradient(&mut self, states: &[MatrixState]) {
        for state in states {
            self.objective
                .exposures
                .observe(self.objective.config.compact_exposures, 2, state);
        }
    }
}

impl FullGraphObjectiveRunner<'_> {
    /// Return the checkpoint identity after binding it to this ordered record
    /// set. A resume with changed data therefore fails contract validation.
    pub fn checkpoint_contract(&self) -> serde_json::Value {
        self.objective.checkpoint_contract()
    }

    /// Snapshot the objective state using the same record-bound contract.
    pub fn checkpoint_state_bytes(&self) -> Result<Vec<u8>, String> {
        self.objective.checkpoint_state_bytes()
    }

    /// Save the trainer and objective as one checkpoint using the runner's
    /// record-bound contract. Callers cannot accidentally save exposure state
    /// under a different data contract.
    pub fn save_checkpoint(
        &self,
        trainer: &mut NeuralTrainer,
        store: &CheckpointStore,
        update_best: bool,
    ) -> Result<CheckpointManifest, TchTrainingError> {
        let objective_state = self
            .checkpoint_state_bytes()
            .map_err(TchTrainingError::Objective)?;
        trainer.set_objective_state(objective_state);
        trainer.save_checkpoint(
            self.variables,
            store,
            self.objective.checkpoint_contract(),
            update_best,
        )
    }
}
