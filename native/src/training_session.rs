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
use crate::objective::{FullGraphObjective, FullGraphObjectiveRunner, StateAdmission};
use crate::training::Progress;
use tch::nn::VarStore;

/// Fixed-size policy for one invocation of the resumable training driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingSessionConfig {
    pub batch_size: usize,
    pub checkpoint_every_attempts: u64,
}

impl TrainingSessionConfig {
    fn validate(self) -> Result<Self, TchTrainingError> {
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

impl<'a> FullGraphTrainingSession<'a> {
    /// Bind and validate a fresh run against a sealed dataset before training.
    pub fn new(
        resources: TrainingSessionResources<'a>,
        config: TrainingSessionConfig,
    ) -> Result<Self, TchTrainingError> {
        let config = config.validate()?;
        let TrainingSessionResources {
            dataset,
            objective,
            model,
            variables,
            registry,
            trainer,
            store,
            admission,
        } = resources;
        validate_dataset_sampler(dataset, trainer)?;
        objective
            .bind_frozen_dataset(dataset)
            .map_err(TchTrainingError::Objective)?;
        FullGraphObjectiveRunner::new(
            objective,
            model,
            variables,
            registry,
            dataset.train.records(),
            admission,
        )
        .map_err(TchTrainingError::Objective)?;

        Ok(Self {
            dataset,
            objective,
            model,
            variables,
            registry,
            trainer,
            store,
            admission,
            config,
        })
    }

    /// Restore model, optimizer, sampler, objective exposures, and caller-owned
    /// RNG bytes before returning a session ready for its next training step.
    pub fn resume_latest(
        resources: TrainingSessionResources<'a>,
        config: TrainingSessionConfig,
    ) -> Result<Self, TchTrainingError> {
        config.validate()?;
        let TrainingSessionResources {
            dataset,
            objective,
            model,
            variables,
            registry,
            trainer,
            store,
            admission,
        } = resources;
        validate_dataset_sampler(dataset, trainer)?;
        objective
            .bind_frozen_dataset(dataset)
            .map_err(TchTrainingError::Objective)?;
        objective.restore_training_checkpoint(
            trainer,
            variables,
            store,
            dataset.train.records(),
        )?;
        Self::new(
            TrainingSessionResources {
                dataset,
                objective,
                model,
                variables,
                registry,
                trainer,
                store,
                admission,
            },
            config,
        )
    }

    /// Execute a bounded number of attempted updates, checkpointing at the
    /// configured interval and always after the final successful attempt.
    /// attempts is not an accepted-update target: rejected non-finite updates
    /// are counted by the trainer and still advance its schedule.
    pub fn run_attempts(
        &mut self,
        attempts: u64,
    ) -> Result<TrainingSessionReport, TchTrainingError> {
        if attempts == 0 {
            return Err(TchTrainingError::Invalid(
                "training session must request at least one attempt",
            ));
        }
        let start_progress = self.trainer.progress();
        let mut last_step = None;
        let mut final_checkpoint = None;
        {
            let mut runner = FullGraphObjectiveRunner::new(
                self.objective,
                self.model,
                self.variables,
                self.registry,
                self.dataset.train.records(),
                self.admission,
            )
            .map_err(TchTrainingError::Objective)?;
            for attempt in 1..=attempts {
                let outcome = self
                    .trainer
                    .step_with_objective(self.config.batch_size, &mut runner)?;
                last_step = Some(outcome);
                if attempt % self.config.checkpoint_every_attempts == 0 || attempt == attempts {
                    final_checkpoint =
                        Some(runner.save_checkpoint(self.trainer, self.store, false)?);
                }
            }
        }
        let end_progress = self.trainer.progress();
        let last_step = last_step.ok_or(TchTrainingError::Invalid(
            "training session completed without an attempted update",
        ))?;
        let final_checkpoint = final_checkpoint.ok_or(TchTrainingError::Invalid(
            "training session completed without a checkpoint",
        ))?;
        if !end_progress.validate()
            || end_progress.attempted.checked_sub(start_progress.attempted) != Some(attempts)
        {
            return Err(TchTrainingError::Invalid(
                "training session progress differs from its attempted-update count",
            ));
        }
        Ok(TrainingSessionReport {
            start_progress,
            end_progress,
            last_step,
            final_checkpoint,
        })
    }

    /// Persist the current state under the objective's frozen-data-bound
    /// checkpoint contract. No dev-best pointer is updated without evaluation.
    pub fn save_checkpoint(&mut self) -> Result<CheckpointManifest, TchTrainingError> {
        let runner = FullGraphObjectiveRunner::new(
            self.objective,
            self.model,
            self.variables,
            self.registry,
            self.dataset.train.records(),
            self.admission,
        )
        .map_err(TchTrainingError::Objective)?;
        runner.save_checkpoint(self.trainer, self.store, false)
    }

    pub fn progress(&self) -> Progress {
        self.trainer.progress()
    }
}

fn validate_dataset_sampler(
    dataset: &FrozenDataset,
    trainer: &NeuralTrainer,
) -> Result<(), TchTrainingError> {
    if dataset.train.is_empty() {
        return Err(TchTrainingError::Invalid(
            "frozen train panel must be nonempty",
        ));
    }
    if trainer.sampler_snapshot().size != dataset.train.len() {
        return Err(TchTrainingError::Invalid(
            "sampler size differs from the pinned frozen train panel",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
