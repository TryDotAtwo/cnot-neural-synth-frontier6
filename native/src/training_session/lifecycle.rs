use super::validation::validate_dataset_sampler;
use super::{FullGraphTrainingSession, TrainingSessionConfig, TrainingSessionResources};
use crate::nn_training::TchTrainingError;
use crate::objective::FullGraphObjectiveRunner;

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
}
