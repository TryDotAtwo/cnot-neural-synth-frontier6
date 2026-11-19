use super::{NeuralTrainer, StepOutcome};
use crate::model::FusionDistanceField;
use crate::nn_training::{optimizer::FlatAdamW, TchTrainingError};
use tch::Tensor;
mod microbatch;
mod data_parallel;

pub struct DeviceUpdateLoss {
    pub loss: Tensor,
    /// Contract/coverage vote. False invalidates the entire interval.
    pub valid: Tensor,
    /// Canonical group participation after OR across every loss term.
    pub group_activity: Tensor,
}

impl NeuralTrainer {
    /// Run one checkpoint interval with CUDA-resident loss, gradients, numerical
    /// admission, AdamW moments and step counters. Sampling uses the bound,
    /// deterministic CPython-compatible host schedule. A failed interval restores
    /// its starting weights and sampler. Invalid AdamW proposals require restore.
    pub fn run_device_updates<F>(
        &mut self,
        model: &FusionDistanceField,
        attempts: u64,
        batch_size: usize,
        mut objective: F,
    ) -> Result<StepOutcome, TchTrainingError>
    where
        F: FnMut(&[usize]) -> Result<DeviceUpdateLoss, String>,
    {
        self.run_device_block(
            model,
            attempts,
            batch_size,
            &mut (),
            |_, indices| objective(indices),
            |_| Ok(()),
        )
    }

    pub(crate) fn run_device_block<S>(
        &mut self,
        model: &FusionDistanceField,
        attempts: u64,
        batch_size: usize,
        state: &mut S,
        mut objective: impl FnMut(&mut S, &[usize]) -> Result<DeviceUpdateLoss, String>,
        commit_audit: impl FnOnce(&mut S) -> Result<(), TchTrainingError>,
    ) -> Result<StepOutcome, TchTrainingError> {
        if self.poisoned {
            return Err(TchTrainingError::Invalid(
                "trainer must restore a checkpoint after a failed optimizer update",
            ));
        }
        if attempts == 0 || attempts > i64::MAX as u64 || batch_size == 0 {
            return Err(TchTrainingError::Invalid("invalid device update interval"));
        }
        let attempted =
            self.progress
                .attempted
                .checked_add(attempts)
                .ok_or(TchTrainingError::Invalid(
                    "training attempt counter overflow",
                ))?;
        let mut flat = tch::no_grad(|| FlatAdamW::new(&self.optimizer, model))?;
        let mut sampler = self.sampler.clone();
        self.poisoned = true;
        let result = (|| {
            for _ in 0..attempts {
                let indices = sampler.next(batch_size)?;
                let data = crate::stage_profile::measure("bellman_and_loss", || {
                    objective(state, &indices)
                })
                .map_err(TchTrainingError::Objective)?;
                flat.validate_loss(&data)?;
                let grads = crate::stage_profile::measure("backward", || {
                    Tensor::f_run_backward(
                        &[&data.loss],
                        &self.optimizer.parameter_refs(),
                        false,
                        false,
                    )
                })?;
                crate::stage_profile::measure("adamw", || {
                    tch::no_grad(|| flat.update(&mut self.optimizer, &data, &grads))
                })?;
            }
            flat.finish(&mut self.optimizer, || commit_audit(state))
        })();
        match result {
            Ok((accepted, last)) => {
                self.progress.attempted = attempted;
                self.progress.accepted += accepted;
                self.progress.rejected += attempts - accepted;
                self.sampler = sampler;
                self.poisoned = false;
                Ok(last)
            }
            Err(error) => {
                tch::no_grad(|| flat.rollback(&mut self.optimizer))?;
                self.poisoned = matches!(
                    error,
                    TchTrainingError::Invalid(crate::nn_training::optimizer::NONFINITE_CANDIDATE)
                        | TchTrainingError::Torch(_)
                );
                Err(error)
            }
        }
    }
}
