use super::super::checkpoint_state::{encode_model_state, validate_model_state};
use super::super::TchTrainingError;
use super::NeuralTrainer;
use crate::checkpoint::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
use crate::training::Sampler;
use serde_json::Value;
use std::io::Cursor;
use tch::nn::VarStore;

impl NeuralTrainer {
    pub fn save_checkpoint(
        &self,
        variables: &VarStore,
        store: &CheckpointStore,
        contract: Value,
        update_best: bool,
    ) -> Result<CheckpointManifest, TchTrainingError> {
        if self.poisoned {
            return Err(TchTrainingError::Invalid(
                "cannot checkpoint a trainer after a failed optimizer update",
            ));
        }
        if self.optimizer.steps() != self.progress.accepted {
            return Err(TchTrainingError::Invalid(
                "AdamW step count differs from accepted update count",
            ));
        }
        let model_state = encode_model_state(variables)?;
        let checkpoint = Checkpoint {
            contract,
            progress: self.progress,
            sampler: self.sampler.snapshot(),
            model_state,
            optimizer_state: self.optimizer.encode()?,
            objective_state: self.objective_state.clone(),
            rng_state: self.rng_state.clone(),
        };
        Ok(store.save(&checkpoint, update_best)?)
    }

    pub fn restore_checkpoint(
        &mut self,
        variables: &mut VarStore,
        store: &CheckpointStore,
        expected_contract: &Value,
    ) -> Result<(), TchTrainingError> {
        let checkpoint = store.load_latest(expected_contract)?;
        self.restore_checkpoint_payload(variables, checkpoint, expected_contract)
    }

    pub(crate) fn restore_checkpoint_payload(
        &mut self,
        variables: &mut VarStore,
        checkpoint: Checkpoint,
        expected_contract: &Value,
    ) -> Result<(), TchTrainingError> {
        if &checkpoint.contract != expected_contract {
            return Err(TchTrainingError::Checkpoint(
                CheckpointError::ContractMismatch,
            ));
        }
        let mut restored_sampler = Sampler::new(checkpoint.sampler.size, 0)?;
        restored_sampler.restore(&checkpoint.sampler)?;
        let decoded_optimizer = self.optimizer.decode(&checkpoint.optimizer_state)?;
        if decoded_optimizer.steps() != checkpoint.progress.accepted {
            return Err(TchTrainingError::Invalid(
                "checkpoint optimizer step count differs from accepted update count",
            ));
        }
        validate_model_state(variables, &checkpoint.model_state)?;

        self.poisoned = true;
        variables.load_from_stream(Cursor::new(checkpoint.model_state))?;
        self.optimizer.apply_decoded(decoded_optimizer);
        self.sampler = restored_sampler;
        self.progress = checkpoint.progress;
        self.objective_state = checkpoint.objective_state;
        self.rng_state = checkpoint.rng_state;
        self.poisoned = false;
        Ok(())
    }
}
