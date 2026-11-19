//! One logical optimizer update with one resident per-example graph at a time.
use super::{DeviceUpdateLoss, FlatAdamW, NeuralTrainer, StepOutcome, TchTrainingError};
use tch::{Kind, Tensor};

impl NeuralTrainer {
    /// Compute each member of the logical batch independently, while retaining
    /// one sample schedule, one averaged gradient and one AdamW update.
    pub fn run_device_microbatch_updates<F>(
        &mut self,
        model: &crate::model::FusionDistanceField,
        attempts: u64,
        batch_size: usize,
        mut objective: F,
    ) -> Result<StepOutcome, TchTrainingError>
    where
        F: FnMut(&[usize], usize) -> Result<DeviceUpdateLoss, String>,
    {
        self.run_device_microbatch_block(
            model,
            attempts,
            batch_size,
            &mut (),
            |_, indices, position| objective(indices, position),
            |_| Ok(()),
        )
    }

    pub(crate) fn run_device_microbatch_block<S>(
        &mut self,
        model: &crate::model::FusionDistanceField,
        attempts: u64,
        batch_size: usize,
        state: &mut S,
        mut objective: impl FnMut(&mut S, &[usize], usize) -> Result<DeviceUpdateLoss, String>,
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
                let mut accumulated: Vec<Option<Tensor>> =
                    (0..self.optimizer.parameter_refs().len())
                        .map(|_| None)
                        .collect();
                let mut losses = Vec::with_capacity(batch_size);
                let mut votes = Vec::with_capacity(batch_size);
                let mut groups = Vec::with_capacity(batch_size);
                for (position, index) in indices.iter().enumerate() {
                    let data = crate::stage_profile::measure("bellman_and_loss", || {
                        objective(state, std::slice::from_ref(index), position)
                    })
                    .map_err(TchTrainingError::Objective)?;
                    // Each real, gradient-bearing loss is checked before its
                    // graph is freed; the final scalar below is detached.
                    flat.validate_loss(&data)?;
                    let scaled = &data.loss / batch_size as f64;
                    let gradients = crate::stage_profile::measure("backward", || {
                        Tensor::f_run_backward(
                            &[&scaled],
                            &self.optimizer.parameter_refs(),
                            false,
                            false,
                        )
                    })?;
                    if gradients.len() != accumulated.len() {
                        return Err(TchTrainingError::Invalid(
                            "autograd gradient count differs from parameter count",
                        ));
                    }
                    tch::no_grad(|| {
                        for (sum, gradient) in accumulated.iter_mut().zip(gradients) {
                            if gradient.defined() {
                                *sum = Some(match sum.take() {
                                    Some(previous) => previous + gradient,
                                    None => gradient,
                                });
                            }
                        }
                    });
                    losses.push(data.loss.detach());
                    votes.push(data.valid);
                    groups.push(data.group_activity);
                }
                let loss = Tensor::stack(&losses, 0).mean(Kind::Float);
                let data = DeviceUpdateLoss {
                    valid: Tensor::stack(&votes, 0).all().logical_and(&loss.isfinite()),
                    group_activity: Tensor::stack(&groups, 0).any_dim(0, false),
                    loss,
                };
                // Missing gradients remain undefined, preserving the existing
                // per-parameter AdamW participation and weight-decay rules.
                let gradients: Vec<_> = accumulated
                    .into_iter()
                    .map(|item| item.unwrap_or_else(Tensor::new))
                    .collect();
                crate::stage_profile::measure("adamw", || {
                    tch::no_grad(|| flat.update(&mut self.optimizer, &data, &gradients))
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
