//! Autograd-backed, resumable training updates for the native Rust model.
//!
//! The objective stays caller-owned: this layer enforces sample order,
//! finite-loss/gradient admission, AdamW state, progress accounting, and a
//! complete immutable checkpoint. It does not claim to implement the CNOT
//! Bellman objective or to provide all 103 feature producers.

use crate::checkpoint::CheckpointError;
use crate::training::TrainingError;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum TchTrainingError {
    Checkpoint(CheckpointError),
    Json(serde_json::Error),
    Sampler(TrainingError),
    Torch(tch::TchError),
    Invalid(&'static str),
    Objective(String),
    Device(String),
}

impl Display for TchTrainingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(error) => Display::fmt(error, f),
            Self::Json(error) => write!(f, "training state metadata is invalid: {error}"),
            Self::Sampler(error) => Display::fmt(error, f),
            Self::Torch(error) => write!(f, "tensor training operation failed: {error}"),
            Self::Invalid(reason) => write!(f, "invalid neural training state: {reason}"),
            Self::Objective(reason) => write!(f, "training objective failed: {reason}"),
            Self::Device(reason) => write!(f, "training device unavailable: {reason}"),
        }
    }
}

impl Error for TchTrainingError {}

impl From<CheckpointError> for TchTrainingError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<serde_json::Error> for TchTrainingError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<TrainingError> for TchTrainingError {
    fn from(value: TrainingError) -> Self {
        Self::Sampler(value)
    }
}

impl From<tch::TchError> for TchTrainingError {
    fn from(value: tch::TchError) -> Self {
        Self::Torch(value)
    }
}

mod checkpoint_state;
mod optimizer;
mod trainer;

pub use optimizer::AdamWConfig;
pub use trainer::{DifferentiableObjective, NeuralTrainer, PreparedTrainingLoss, StepOutcome};
