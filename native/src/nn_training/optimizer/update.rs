use super::super::TchTrainingError;
use super::state::AdamW;
use tch::Tensor;

struct AdamWUpdate {
    parameter: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
}

impl AdamW {
    pub(in crate::nn_training) fn gradient_status(
        &self,
        gradients: &[Tensor],
    ) -> Result<(bool, bool), TchTrainingError> {
        if gradients.len() != self.parameters.len() {
            return Err(TchTrainingError::Invalid(
                "autograd gradient count differs from trainable parameter count",
            ));
        }
        let mut any = false;
        for gradient in gradients {
            if !gradient.defined() {
                continue;
            }
            any = true;
            if gradient.is_sparse() {
                return Err(TchTrainingError::Invalid(
                    "AdamW does not support sparse gradients",
                ));
            }
            if gradient.isfinite().all().int64_value(&[]) == 0 {
                return Ok((true, false));
            }
        }
        Ok((any, true))
    }

    pub(in crate::nn_training) fn step(
        &mut self,
        gradients: &[Tensor],
    ) -> Result<(), TchTrainingError> {
        if gradients.len() != self.parameters.len() {
            return Err(TchTrainingError::Invalid(
                "autograd gradient count differs from trainable parameter count",
            ));
        }
        let next_step = self
            .steps
            .checked_add(1)
            .ok_or(TchTrainingError::Invalid("optimizer step counter overflow"))?;
        let next_parameter_steps: Vec<_> = self
            .parameters
            .iter()
            .zip(gradients)
            .map(|(state, gradient)| {
                if gradient.defined() {
                    state.steps.checked_add(1).ok_or(TchTrainingError::Invalid(
                        "parameter optimizer step counter overflow",
                    ))
                } else {
                    Ok(state.steps)
                }
            })
            .collect::<Result<_, _>>()?;
        let learning_rate = self.config.learning_rate;
        let decay_factor = 1.0 - learning_rate * self.config.weight_decay;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let epsilon = self.config.epsilon;

        let updates = tch::no_grad(|| -> Result<Vec<Option<AdamWUpdate>>, TchTrainingError> {
            let mut updates = Vec::with_capacity(self.parameters.len());
            for ((state, gradient), next_parameter_step) in self
                .parameters
                .iter()
                .zip(gradients)
                .zip(&next_parameter_steps)
            {
                if !gradient.defined() {
                    updates.push(None);
                    continue;
                }
                let beta1_correction = 1.0 - beta1.powf(*next_parameter_step as f64);
                let beta2_correction = 1.0 - beta2.powf(*next_parameter_step as f64);
                let first = &state.first_moment * beta1 + gradient * (1.0 - beta1);
                let second =
                    &state.second_moment * beta2 + gradient.pow_tensor_scalar(2.0) * (1.0 - beta2);
                let denominator = second.sqrt() / beta2_correction.sqrt() + epsilon;
                let update = &first * (learning_rate / beta1_correction) / denominator;
                let parameter = if self.config.weight_decay != 0.0 {
                    &state.parameter * decay_factor - &update
                } else {
                    &state.parameter - &update
                };
                if first.isfinite().all().int64_value(&[]) == 0
                    || second.isfinite().all().int64_value(&[]) == 0
                    || update.isfinite().all().int64_value(&[]) == 0
                    || parameter.isfinite().all().int64_value(&[]) == 0
                {
                    return Err(TchTrainingError::Invalid(
                        "AdamW candidate state contains non-finite values",
                    ));
                }
                updates.push(Some(AdamWUpdate {
                    parameter,
                    first_moment: first,
                    second_moment: second,
                }));
            }
            Ok(updates)
        })?;

        tch::no_grad(|| -> Result<(), TchTrainingError> {
            for (state, update) in self.parameters.iter_mut().zip(updates) {
                let Some(update) = update else {
                    continue;
                };
                state.first_moment.f_copy_(&update.first_moment)?;
                state.second_moment.f_copy_(&update.second_moment)?;
                state.parameter.f_copy_(&update.parameter)?;
            }
            Ok(())
        })?;
        self.steps = next_step;
        for (state, next_parameter_step) in self.parameters.iter_mut().zip(next_parameter_steps) {
            state.steps = next_parameter_step;
        }
        Ok(())
    }
}
