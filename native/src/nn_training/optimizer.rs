use super::TchTrainingError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use tch::nn::VarStore;
use tch::{Device, Tensor};

const OPTIMIZER_MAGIC: &[u8; 8] = b"C103ADW1";
const OPTIMIZER_VERSION: u32 = 1;
const MAX_OPTIMIZER_HEADER_BYTES: usize = 16 * 1024 * 1024;

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
    fn validate(self) -> Result<(), TchTrainingError> {
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

#[derive(Debug)]
struct ParameterState {
    name: String,
    parameter: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    steps: u64,
}

struct AdamWUpdate {
    parameter: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
}

#[derive(Debug)]
pub(super) struct AdamW {
    config: AdamWConfig,
    steps: u64,
    parameters: Vec<ParameterState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterSignature {
    name: String,
    shape: Vec<i64>,
    kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdamWHeader {
    version: u32,
    config: AdamWConfig,
    steps: u64,
    parameter_steps: Vec<u64>,
    parameters: Vec<ParameterSignature>,
}

pub(super) struct DecodedAdamW {
    steps: u64,
    parameter_steps: Vec<u64>,
    moments: Vec<(Tensor, Tensor)>,
}

impl AdamW {
    pub(super) fn new(variables: &VarStore, config: AdamWConfig) -> Result<Self, TchTrainingError> {
        config.validate()?;
        let mut named: Vec<_> = variables
            .variables()
            .into_iter()
            .filter(|(_, tensor)| tensor.requires_grad())
            .collect();
        named.sort_by(|left, right| left.0.cmp(&right.0));
        if named.is_empty() {
            return Err(TchTrainingError::Invalid(
                "model has no trainable parameters",
            ));
        }
        let parameters = named
            .into_iter()
            .map(|(name, parameter)| ParameterState {
                name,
                first_moment: Tensor::zeros_like(&parameter),
                second_moment: Tensor::zeros_like(&parameter),
                steps: 0,
                parameter,
            })
            .collect();
        Ok(Self {
            config,
            steps: 0,
            parameters,
        })
    }

    pub(super) fn steps(&self) -> u64 {
        self.steps
    }

    pub(super) fn parameter_refs(&self) -> Vec<&Tensor> {
        self.parameters
            .iter()
            .map(|state| &state.parameter)
            .collect()
    }

    pub(super) fn gradient_status(
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

    pub(super) fn step(&mut self, gradients: &[Tensor]) -> Result<(), TchTrainingError> {
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

    fn signatures(&self) -> Vec<ParameterSignature> {
        self.parameters
            .iter()
            .map(|state| ParameterSignature {
                name: state.name.clone(),
                shape: state.parameter.size(),
                kind: format!("{:?}", state.parameter.kind()),
            })
            .collect()
    }

    pub(super) fn encode(&self) -> Result<Vec<u8>, TchTrainingError> {
        let header = AdamWHeader {
            version: OPTIMIZER_VERSION,
            config: self.config,
            steps: self.steps,
            parameter_steps: self.parameters.iter().map(|state| state.steps).collect(),
            parameters: self.signatures(),
        };
        let header_bytes = serde_json::to_vec(&header)?;
        if header_bytes.len() > MAX_OPTIMIZER_HEADER_BYTES {
            return Err(TchTrainingError::Invalid(
                "AdamW metadata header exceeds size limit",
            ));
        }

        let mut named_tensors = Vec::with_capacity(self.parameters.len() * 2);
        for (index, state) in self.parameters.iter().enumerate() {
            named_tensors.push((
                format!("m_{index:08}"),
                state.first_moment.detach().to_device(Device::Cpu),
            ));
            named_tensors.push((
                format!("v_{index:08}"),
                state.second_moment.detach().to_device(Device::Cpu),
            ));
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(OPTIMIZER_MAGIC);
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        Tensor::save_multi_to_stream(&named_tensors, &mut bytes)?;
        Ok(bytes)
    }

    pub(super) fn decode(&self, bytes: &[u8]) -> Result<DecodedAdamW, TchTrainingError> {
        if bytes.len() < OPTIMIZER_MAGIC.len() + 8 || &bytes[..8] != OPTIMIZER_MAGIC {
            return Err(TchTrainingError::Invalid("AdamW state magic differs"));
        }
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        if header_len > MAX_OPTIMIZER_HEADER_BYTES {
            return Err(TchTrainingError::Invalid(
                "AdamW metadata header exceeds size limit",
            ));
        }
        let payload_start = 16usize
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or(TchTrainingError::Invalid("AdamW state is truncated"))?;
        let header: AdamWHeader = serde_json::from_slice(&bytes[16..payload_start])?;
        if header.version != OPTIMIZER_VERSION
            || header.config != self.config
            || header.parameters != self.signatures()
            || header.parameter_steps.len() != self.parameters.len()
            || header
                .parameter_steps
                .iter()
                .any(|steps| *steps > header.steps)
        {
            return Err(TchTrainingError::Invalid(
                "AdamW state is incompatible with this model or configuration",
            ));
        }
        let loaded = Tensor::load_multi_from_stream_with_device(
            Cursor::new(&bytes[payload_start..]),
            self.parameters[0].parameter.device(),
        )?;
        let tensor_map: HashMap<_, _> = loaded.into_iter().collect();
        if tensor_map.len() != self.parameters.len() * 2 {
            return Err(TchTrainingError::Invalid(
                "AdamW state tensor count differs",
            ));
        }

        let mut moments = Vec::with_capacity(self.parameters.len());
        for (index, state) in self.parameters.iter().enumerate() {
            let first = tensor_map
                .get(&format!("m_{index:08}"))
                .ok_or(TchTrainingError::Invalid("AdamW first moment is missing"))?;
            let second = tensor_map
                .get(&format!("v_{index:08}"))
                .ok_or(TchTrainingError::Invalid("AdamW second moment is missing"))?;
            if first.size() != state.parameter.size()
                || second.size() != state.parameter.size()
                || first.kind() != state.parameter.kind()
                || second.kind() != state.parameter.kind()
                || first.isfinite().all().int64_value(&[]) == 0
                || second.isfinite().all().int64_value(&[]) == 0
            {
                return Err(TchTrainingError::Invalid(
                    "AdamW state tensor shape, type, or value differs",
                ));
            }
            moments.push((first.shallow_clone(), second.shallow_clone()));
        }
        Ok(DecodedAdamW {
            steps: header.steps,
            parameter_steps: header.parameter_steps,
            moments,
        })
    }

    pub(super) fn apply_decoded(&mut self, decoded: DecodedAdamW) {
        self.steps = decoded.steps;
        for ((state, (first, second)), steps) in self
            .parameters
            .iter_mut()
            .zip(decoded.moments)
            .zip(decoded.parameter_steps)
        {
            state.first_moment = first;
            state.second_moment = second;
            state.steps = steps;
        }
    }
}

impl DecodedAdamW {
    pub(super) fn steps(&self) -> u64 {
        self.steps
    }
}
