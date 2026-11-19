use super::{FullGraphTrainingSession, TrainingSessionReport};
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::model::FusionDistanceField;
use crate::nn_training::TchTrainingError;
use crate::objective::FullGraphObjectiveRunner;
use std::collections::HashMap;
use tch::nn::VarStore;

impl FullGraphTrainingSession<'_> {
    /// Use the same pinned session/checkpoint lifecycle, with the resident
    /// primitive objective and optimizer for every update between checkpoints.
    pub fn run_device_attempts(
        &mut self,
        attempts: u64,
        plans: HashMap<u32, DeviceFeaturePlan>,
    ) -> Result<TrainingSessionReport, TchTrainingError> {
        if attempts == 0 {
            return Err(TchTrainingError::Invalid(
                "training session must request at least one attempt",
            ));
        }
        let start_progress = self.trainer.progress();
        let mut runner = crate::stage_profile::measure("prepare_records_and_layouts", || {
            FullGraphObjectiveRunner::new(
                self.objective,
                self.model,
                self.variables,
                self.registry,
                self.dataset.train.records(),
                self.admission,
            )
            .and_then(|r| r.into_device(plans))
        })
        .map_err(TchTrainingError::Objective)?;
        let mut remaining = attempts;
        loop {
            let count = remaining.min(self.config.checkpoint_every_attempts);
            let last_step = crate::stage_profile::measure("interval_bookkeeping", || {
                runner.run_interval(self.trainer, count, self.config.batch_size)
            })?;
            let final_checkpoint = crate::stage_profile::measure("checkpoint_write", || {
                runner
                    .source
                    .save_checkpoint(self.trainer, self.store, false)
            })?;
            remaining -= count;
            if remaining == 0 {
                return Ok(TrainingSessionReport {
                    start_progress,
                    end_progress: self.trainer.progress(),
                    last_step,
                    final_checkpoint,
                });
            }
        }
    }

    /// Train with two CUDA replicas while keeping one sampler, optimizer, and
    /// checkpoint owner on the master device.
    pub fn run_device_data_parallel_attempts(
        &mut self,
        attempts: u64,
        plans: HashMap<u32, DeviceFeaturePlan>,
        replica_model: &FusionDistanceField,
        replica_variables: &mut VarStore,
        replica_plans: HashMap<u32, DeviceFeaturePlan>,
    ) -> Result<TrainingSessionReport, TchTrainingError> {
        if attempts == 0 {
            return Err(TchTrainingError::Invalid(
                "training session must request at least one attempt",
            ));
        }
        let start_progress = self.trainer.progress();
        let mut runner = crate::stage_profile::measure("prepare_records_and_layouts", || {
            FullGraphObjectiveRunner::new(
                self.objective,
                self.model,
                self.variables,
                self.registry,
                self.dataset.train.records(),
                self.admission,
            )
            .and_then(|r| {
                r.into_data_parallel(plans, replica_model, replica_variables, replica_plans)
            })
        })
        .map_err(TchTrainingError::Objective)?;
        let mut remaining = attempts;
        loop {
            let count = remaining.min(self.config.checkpoint_every_attempts);
            let last_step = crate::stage_profile::measure("interval_bookkeeping", || {
                runner.run_interval(self.trainer, count, self.config.batch_size)
            })?;
            let final_checkpoint = crate::stage_profile::measure("checkpoint_write", || {
                runner
                    .source
                    .save_checkpoint(self.trainer, self.store, false)
            })?;
            remaining -= count;
            if remaining == 0 {
                return Ok(TrainingSessionReport {
                    start_progress,
                    end_progress: self.trainer.progress(),
                    last_step,
                    final_checkpoint,
                });
            }
        }
    }
}
