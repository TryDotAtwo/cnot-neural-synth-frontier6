use super::super::TchTrainingError;
use super::{DifferentiableObjective, NeuralTrainer, PreparedTrainingLoss, StepOutcome};
use crate::state::MatrixState;
use tch::Tensor;

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

impl NeuralTrainer {
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
}
