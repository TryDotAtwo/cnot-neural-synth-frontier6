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
                    return Err(TchTrainingError::Invalid(super::NONFINITE_CANDIDATE));
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

#[cfg(test)]
mod inactive_parameter_tests {
    use super::super::config::AdamWConfig;
    use super::super::state::AdamW;
    use tch::nn::{Init, VarStore};
    use tch::{Device, Tensor};

    #[test]
    fn undefined_inactive_parameter_keeps_weight_moments_and_bias_step_unchanged() {
        let variables = VarStore::new(Device::Cpu);
        let root = variables.root();
        let _active = root.var("active_group", &[1], Init::Const(1.0));
        let _inactive = root.var("inactive_group", &[1], Init::Const(2.0));
        let config = AdamWConfig {
            learning_rate: 0.1,
            beta1: 0.5,
            beta2: 0.5,
            epsilon: 1e-8,
            weight_decay: 0.25,
        };
        let mut optimizer = AdamW::new(&variables, config).unwrap();
        let active_index = optimizer
            .parameters
            .iter()
            .position(|state| state.name == "active_group")
            .unwrap();
        let inactive_index = optimizer
            .parameters
            .iter()
            .position(|state| state.name == "inactive_group")
            .unwrap();
        let inactive_parameter = optimizer.parameters[inactive_index]
            .parameter
            .detach()
            .copy();
        let inactive_first = optimizer.parameters[inactive_index]
            .first_moment
            .detach()
            .copy();
        let inactive_second = optimizer.parameters[inactive_index]
            .second_moment
            .detach()
            .copy();

        let mut gradients: Vec<_> = optimizer
            .parameters
            .iter()
            .map(|_| Tensor::new())
            .collect();
        gradients[active_index] =
            Tensor::ones_like(&optimizer.parameters[active_index].parameter);
        optimizer.step(&gradients).unwrap();

        assert_eq!(optimizer.steps(), 1);
        assert_eq!(optimizer.parameters[active_index].steps, 1);
        assert_eq!(optimizer.parameters[inactive_index].steps, 0);
        assert!(optimizer.parameters[inactive_index]
            .parameter
            .equal(&inactive_parameter));
        assert!(optimizer.parameters[inactive_index]
            .first_moment
            .equal(&inactive_first));
        assert!(optimizer.parameters[inactive_index]
            .second_moment
            .equal(&inactive_second));

        let gradients: Vec<_> = optimizer
            .parameters
            .iter()
            .map(|state| Tensor::ones_like(&state.parameter))
            .collect();
        optimizer.step(&gradients).unwrap();

        assert_eq!(optimizer.steps(), 2);
        assert_eq!(optimizer.parameters[inactive_index].steps, 1);
        let resumed = optimizer.parameters[inactive_index].parameter.double_value(&[0]);
        assert!((resumed - 1.85).abs() < 1e-5, "inactive parameter={resumed}");
        assert!(
            (optimizer.parameters[inactive_index]
                .first_moment
                .double_value(&[0])
                - 0.5)
                .abs()
                < 1e-6
        );
        assert!(
            (optimizer.parameters[inactive_index]
                .second_moment
                .double_value(&[0])
                - 0.5)
                .abs()
                < 1e-6
        );
        assert_eq!(optimizer.parameters[active_index].steps, 2);
    }
}
