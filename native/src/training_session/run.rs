use super::{FullGraphTrainingSession, TrainingSessionReport};
use crate::checkpoint::CheckpointManifest;
use crate::nn_training::TchTrainingError;
use crate::objective::FullGraphObjectiveRunner;
use crate::training::Progress;

impl<'a> FullGraphTrainingSession<'a> {
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
