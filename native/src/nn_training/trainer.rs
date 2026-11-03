use super::checkpoint_state::{encode_model_state, validate_model_state};
use super::optimizer::AdamW;
use super::{AdamWConfig, TchTrainingError};
use crate::checkpoint::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
use crate::state::MatrixState;
use crate::training::{Progress, Sampler, SamplerSnapshot};
use serde_json::Value;
use std::io::Cursor;
use tch::nn::VarStore;
use tch::Tensor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    pub accepted: bool,
    pub loss: f64,
}

/// A scalar loss and the states whose autograd graph contributes to it.
pub struct PreparedTrainingLoss {
    pub loss: Tensor,
    pub gradient_states: Vec<MatrixState>,
}

/// Couples loss construction to the actual autograd boundary so exposure
/// ledgers are updated only when backward is attempted.
pub trait DifferentiableObjective {
    fn compute(&mut self, indices: &[usize]) -> Result<PreparedTrainingLoss, String>;

    fn observe_gradient(&mut self, states: &[MatrixState]);
}

struct ClosureObjective<F>(Option<F>);

impl<F> DifferentiableObjective for ClosureObjective<F>
where
    F: FnOnce(&[usize]) -> Result<Tensor, String>,
{
    fn compute(&mut self, indices: &[usize]) -> Result<PreparedTrainingLoss, String> {
        let objective = self
            .0
            .take()
            .ok_or_else(|| "training closure was evaluated more than once".to_owned())?;
        Ok(PreparedTrainingLoss {
            loss: objective(indices)?,
            gradient_states: Vec::new(),
        })
    }

    fn observe_gradient(&mut self, _states: &[MatrixState]) {}
}

/// Owns the sample schedule and optimizer counters for one model training run.
pub struct NeuralTrainer {
    optimizer: AdamW,
    sampler: Sampler,
    progress: Progress,
    objective_state: Vec<u8>,
    rng_state: Vec<u8>,
    poisoned: bool,
}

impl NeuralTrainer {
    pub fn new(
        variables: &VarStore,
        sampler: Sampler,
        optimizer: AdamWConfig,
    ) -> Result<Self, TchTrainingError> {
        crate::torch_device::require_device(variables.device())
            .map_err(TchTrainingError::Device)?;
        Ok(Self {
            optimizer: AdamW::new(variables, optimizer)?,
            sampler,
            progress: Progress::default(),
            objective_state: Vec::new(),
            rng_state: Vec::new(),
            poisoned: false,
        })
    }

    /// Compute one scalar loss and apply AdamW only when loss and every present
    /// gradient are finite. Rejected numerical updates advance the sampler and
    /// attempted/rejected counters, but leave parameters and optimizer intact.
    pub fn step<F>(
        &mut self,
        batch_size: usize,
        objective: F,
    ) -> Result<StepOutcome, TchTrainingError>
    where
        F: FnOnce(&[usize]) -> Result<Tensor, String>,
    {
        let mut objective = ClosureObjective(Some(objective));
        self.step_with_objective(batch_size, &mut objective)
    }

    /// Run a loss and commit its staged gradient exposures at the point where
    /// autograd is actually invoked. If autograd errors part-way through, the
    /// exposure set is still conservatively committed before returning.
    pub fn step_with_objective<O>(
        &mut self,
        batch_size: usize,
        objective: &mut O,
    ) -> Result<StepOutcome, TchTrainingError>
    where
        O: DifferentiableObjective + ?Sized,
    {
        if self.poisoned {
            return Err(TchTrainingError::Invalid(
                "trainer must restore a checkpoint after a failed optimizer update",
            ));
        }
        let sampler_before = self.sampler.snapshot();
        let indices = self.sampler.next(batch_size)?;
        let prepared = match objective.compute(&indices) {
            Ok(prepared) => prepared,
            Err(reason) => {
                self.sampler.restore(&sampler_before)?;
                return Err(TchTrainingError::Objective(reason));
            }
        };
        let loss = prepared.loss;
        if loss.numel() != 1 {
            self.sampler.restore(&sampler_before)?;
            return Err(TchTrainingError::Invalid(
                "objective must return exactly one scalar loss",
            ));
        }
        if !loss.requires_grad() {
            self.sampler.restore(&sampler_before)?;
            return Err(TchTrainingError::Invalid(
                "objective loss is not attached to an autograd graph",
            ));
        }
        let loss_value = loss.double_value(&[]);
        if !loss_value.is_finite() {
            let mut progress = self.progress;
            progress.record_attempt(false)?;
            self.progress = progress;
            return Ok(StepOutcome {
                accepted: false,
                loss: loss_value,
            });
        }

        let parameters = self.optimizer.parameter_refs();
        let gradients = match Tensor::f_run_backward(&[&loss], &parameters, false, false) {
            Ok(gradients) => {
                objective.observe_gradient(&prepared.gradient_states);
                gradients
            }
            Err(error) => {
                objective.observe_gradient(&prepared.gradient_states);
                self.sampler.restore(&sampler_before)?;
                return Err(TchTrainingError::Torch(error));
            }
        };
        let (any_gradient, finite_gradients) = self.optimizer.gradient_status(&gradients)?;
        if !any_gradient {
            self.sampler.restore(&sampler_before)?;
            return Err(TchTrainingError::Invalid(
                "objective produced no gradients for trainable parameters",
            ));
        }
        if !finite_gradients {
            let mut progress = self.progress;
            progress.record_attempt(false)?;
            self.progress = progress;
            return Ok(StepOutcome {
                accepted: false,
                loss: loss_value,
            });
        }

        let mut progress = self.progress;
        progress.record_attempt(true)?;
        if let Err(error) = self.optimizer.step(&gradients) {
            self.poisoned = true;
            return Err(error);
        }
        self.progress = progress;
        Ok(StepOutcome {
            accepted: true,
            loss: loss_value,
        })
    }

    pub fn progress(&self) -> Progress {
        self.progress
    }

    pub fn optimizer_steps(&self) -> u64 {
        self.optimizer.steps()
    }

    pub fn sampler_snapshot(&self) -> SamplerSnapshot {
        self.sampler.snapshot()
    }

    pub fn set_objective_state(&mut self, state: Vec<u8>) {
        self.objective_state = state;
    }

    pub fn objective_state(&self) -> &[u8] {
        &self.objective_state
    }

    /// Store caller-managed random-generator bytes. The caller must restore
    /// the generator from `rng_state()` before computing another objective.
    pub fn set_rng_state(&mut self, state: Vec<u8>) {
        self.rng_state = state;
    }

    pub fn rng_state(&self) -> &[u8] {
        &self.rng_state
    }

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
