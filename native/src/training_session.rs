//! Resumable end-to-end orchestration for the verified full-graph training path.
//!
//! This module deliberately does not invent a feature profile or physical
//! contexts. Callers provide the registry, model, objective, and a sealed
//! FrozenDataset; the session binds that exact artifact and its train panel
//! into the objective/checkpoint contract before the first update.

use crate::checkpoint::{CheckpointManifest, CheckpointStore};
use crate::features::FeatureRegistry;
use crate::frozen_data::FrozenDataset;
use crate::model::FusionDistanceField;
use crate::nn_training::{NeuralTrainer, StepOutcome, TchTrainingError};
use crate::objective::{FullGraphObjective, StateAdmission};
use crate::training::Progress;
use tch::nn::VarStore;

/// Fixed-size policy for one invocation of the resumable training driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingSessionConfig {
    pub batch_size: usize,
    pub checkpoint_every_attempts: u64,
}

impl TrainingSessionConfig {
    pub(super) fn validate(self) -> Result<Self, TchTrainingError> {
        if self.batch_size == 0 {
            return Err(TchTrainingError::Invalid(
                "training session batch size must be positive",
            ));
        }
        if self.checkpoint_every_attempts == 0 {
            return Err(TchTrainingError::Invalid(
                "training session checkpoint interval must be positive",
            ));
        }
        Ok(self)
    }
}

/// The durable result of a bounded session call.
#[derive(Clone, Debug)]
pub struct TrainingSessionReport {
    pub start_progress: Progress,
    pub end_progress: Progress,
    pub last_step: StepOutcome,
    pub final_checkpoint: CheckpointManifest,
}

/// Borrowed inputs shared by a resumable training session.
pub struct TrainingSessionResources<'a> {
    pub dataset: &'a FrozenDataset,
    pub objective: &'a mut FullGraphObjective,
    pub model: &'a FusionDistanceField,
    pub variables: &'a mut VarStore,
    pub registry: &'a FeatureRegistry,
    pub trainer: &'a mut NeuralTrainer,
    pub store: &'a CheckpointStore,
    pub admission: Option<&'a dyn StateAdmission>,
}

/// Owns the mutable training and objective state for one verified frozen train
/// panel. A session can run additional attempts, save, or resume the latest
/// checkpoint whose contract matches the pinned artifact and exact train rows.
pub struct FullGraphTrainingSession<'a> {
    dataset: &'a FrozenDataset,
    objective: &'a mut FullGraphObjective,
    model: &'a FusionDistanceField,
    variables: &'a mut VarStore,
    registry: &'a FeatureRegistry,
    trainer: &'a mut NeuralTrainer,
    store: &'a CheckpointStore,
    admission: Option<&'a dyn StateAdmission>,
    config: TrainingSessionConfig,
}

mod lifecycle;
mod run;
mod validation;

#[cfg(test)]
mod tests;
