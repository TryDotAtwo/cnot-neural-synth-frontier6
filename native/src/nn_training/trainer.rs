use super::optimizer::AdamW;
use super::{AdamWConfig, TchTrainingError};
use crate::state::MatrixState;
use crate::training::{Progress, Sampler, SamplerSnapshot};
use tch::nn::VarStore;
use tch::Tensor;

mod checkpoint;
mod step;

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
}
