use super::super::TchTrainingError;
use serde::{Deserialize, Serialize};
/// AdamW parameters, matching `torch.optim.AdamW` defaults except for the
/// deliberately explicit learning rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdamWConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl AdamWConfig {
    pub(super) fn validate(self) -> Result<(), TchTrainingError> {
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.beta1.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
            || self.learning_rate * self.weight_decay >= 1.0
        {
            return Err(TchTrainingError::Invalid(
                "AdamW hyperparameters are outside their valid ranges",
            ));
        }
        Ok(())
    }
}
