//! Autograd-backed, resumable training updates for the native Rust model.
//!
//! The objective stays caller-owned: this layer enforces sample order,
//! finite-loss/gradient admission, AdamW state, progress accounting, and a
//! complete immutable checkpoint. It does not claim to implement the CNOT
//! Bellman objective or to provide all 103 feature producers.

use crate::checkpoint::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
use crate::state::MatrixState;
use crate::training::{Progress, Sampler, SamplerSnapshot, TrainingError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::Cursor;
use tch::nn::VarStore;
use tch::{Device, Tensor};

const OPTIMIZER_MAGIC: &[u8; 8] = b"C103ADW1";
const OPTIMIZER_VERSION: u32 = 1;
const MAX_OPTIMIZER_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub enum TchTrainingError {
    Checkpoint(CheckpointError),
    Json(serde_json::Error),
    Sampler(TrainingError),
    Torch(tch::TchError),
    Invalid(&'static str),
    Objective(String),
    Device(String),
}

impl Display for TchTrainingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(error) => Display::fmt(error, f),
            Self::Json(error) => write!(f, "training state metadata is invalid: {error}"),
            Self::Sampler(error) => Display::fmt(error, f),
            Self::Torch(error) => write!(f, "tensor training operation failed: {error}"),
            Self::Invalid(reason) => write!(f, "invalid neural training state: {reason}"),
            Self::Objective(reason) => write!(f, "training objective failed: {reason}"),
            Self::Device(reason) => write!(f, "training device unavailable: {reason}"),
        }
    }
}

impl Error for TchTrainingError {}

impl From<CheckpointError> for TchTrainingError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<serde_json::Error> for TchTrainingError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<TrainingError> for TchTrainingError {
    fn from(value: TrainingError) -> Self {
        Self::Sampler(value)
    }
}

impl From<tch::TchError> for TchTrainingError {
    fn from(value: tch::TchError) -> Self {
        Self::Torch(value)
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    pub accepted: bool,
    pub loss: f64,
}

/// A scalar loss and the states whose autograd graph contributes to it.
pub struct PreparedTrainingLoss {
    pub loss: Tensor,
    pub gradient_states: Vec<MatrixState>,
}

/// Couples loss construction to the actual autograd boundary so exposure
/// ledgers are updated only when backward is attempted.
pub trait DifferentiableObjective {
    fn compute(&mut self, indices: &[usize]) -> Result<PreparedTrainingLoss, String>;

    fn observe_gradient(&mut self, states: &[MatrixState]);
}

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
struct AdamW {
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

struct DecodedAdamW {
    steps: u64,
    parameter_steps: Vec<u64>,
    moments: Vec<(Tensor, Tensor)>,
}

impl AdamW {
    fn new(variables: &VarStore, config: AdamWConfig) -> Result<Self, TchTrainingError> {
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

    fn gradient_status(&self, gradients: &[Tensor]) -> Result<(bool, bool), TchTrainingError> {
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

    fn step(&mut self, gradients: &[Tensor]) -> Result<(), TchTrainingError> {
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

    fn encode(&self) -> Result<Vec<u8>, TchTrainingError> {
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

    fn decode(&self, bytes: &[u8]) -> Result<DecodedAdamW, TchTrainingError> {
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

    fn apply_decoded(&mut self, decoded: DecodedAdamW) {
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

/// Owns the sample schedule and optimizer counters for one model training run.
pub struct NeuralTrainer {
    optimizer: AdamW,
    sampler: Sampler,
    progress: Progress,
    objective_state: Vec<u8>,
    rng_state: Vec<u8>,
    poisoned: bool,
}

impl NeuralTrainer {
    pub fn new(
        variables: &VarStore,
        sampler: Sampler,
        optimizer: AdamWConfig,
    ) -> Result<Self, TchTrainingError> {
        crate::torch_device::require_device(variables.device())
            .map_err(TchTrainingError::Device)?;
        Ok(Self {
            optimizer: AdamW::new(variables, optimizer)?,
            sampler,
            progress: Progress::default(),
            objective_state: Vec::new(),
            rng_state: Vec::new(),
            poisoned: false,
        })
    }

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

        let parameters: Vec<_> = self
            .optimizer
            .parameters
            .iter()
            .map(|parameter| &parameter.parameter)
            .collect();
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

    pub fn progress(&self) -> Progress {
        self.progress
    }

    pub fn optimizer_steps(&self) -> u64 {
        self.optimizer.steps
    }

    pub fn sampler_snapshot(&self) -> SamplerSnapshot {
        self.sampler.snapshot()
    }

    pub fn set_objective_state(&mut self, state: Vec<u8>) {
        self.objective_state = state;
    }

    pub fn objective_state(&self) -> &[u8] {
        &self.objective_state
    }

    /// Store caller-managed random-generator bytes. The caller must restore
    /// the generator from `rng_state()` before computing another objective.
    pub fn set_rng_state(&mut self, state: Vec<u8>) {
        self.rng_state = state;
    }

    pub fn rng_state(&self) -> &[u8] {
        &self.rng_state
    }

    pub fn save_checkpoint(
        &self,
        variables: &VarStore,
        store: &CheckpointStore,
        contract: Value,
        update_best: bool,
    ) -> Result<CheckpointManifest, TchTrainingError> {
        if self.poisoned {
            return Err(TchTrainingError::Invalid(
                "cannot checkpoint a trainer after a failed optimizer update",
            ));
        }
        if self.optimizer.steps != self.progress.accepted {
            return Err(TchTrainingError::Invalid(
                "AdamW step count differs from accepted update count",
            ));
        }
        let model_state = encode_model_state(variables)?;
        let checkpoint = Checkpoint {
            contract,
            progress: self.progress,
            sampler: self.sampler.snapshot(),
            model_state,
            optimizer_state: self.optimizer.encode()?,
            objective_state: self.objective_state.clone(),
            rng_state: self.rng_state.clone(),
        };
        Ok(store.save(&checkpoint, update_best)?)
    }

    pub fn restore_checkpoint(
        &mut self,
        variables: &mut VarStore,
        store: &CheckpointStore,
        expected_contract: &Value,
    ) -> Result<(), TchTrainingError> {
        let checkpoint = store.load_latest(expected_contract)?;
        self.restore_checkpoint_payload(variables, checkpoint, expected_contract)
    }

    pub(crate) fn restore_checkpoint_payload(
        &mut self,
        variables: &mut VarStore,
        checkpoint: Checkpoint,
        expected_contract: &Value,
    ) -> Result<(), TchTrainingError> {
        if &checkpoint.contract != expected_contract {
            return Err(TchTrainingError::Checkpoint(
                CheckpointError::ContractMismatch,
            ));
        }
        let mut restored_sampler = Sampler::new(checkpoint.sampler.size, 0)?;
        restored_sampler.restore(&checkpoint.sampler)?;
        let decoded_optimizer = self.optimizer.decode(&checkpoint.optimizer_state)?;
        if decoded_optimizer.steps != checkpoint.progress.accepted {
            return Err(TchTrainingError::Invalid(
                "checkpoint optimizer step count differs from accepted update count",
            ));
        }
        validate_model_state(variables, &checkpoint.model_state)?;

        self.poisoned = true;
        variables.load_from_stream(Cursor::new(checkpoint.model_state))?;
        self.optimizer.apply_decoded(decoded_optimizer);
        self.sampler = restored_sampler;
        self.progress = checkpoint.progress;
        self.objective_state = checkpoint.objective_state;
        self.rng_state = checkpoint.rng_state;
        self.poisoned = false;
        Ok(())
    }
}

fn encode_model_state(variables: &VarStore) -> Result<Vec<u8>, TchTrainingError> {
    // VarStore::save_to_stream iterates its internal HashMap. Sort names first
    // so logically identical checkpoints have identical model payload bytes and
    // therefore stable content hashes across process reconstructions.
    let mut named_tensors: Vec<_> = variables.variables().into_iter().collect();
    named_tensors.sort_by(|left, right| left.0.cmp(&right.0));
    let mut bytes = Vec::new();
    Tensor::save_multi_to_stream(&named_tensors, &mut bytes)?;
    Ok(bytes)
}

fn validate_model_state(variables: &VarStore, bytes: &[u8]) -> Result<(), TchTrainingError> {
    let loaded =
        Tensor::load_multi_from_stream_with_device(Cursor::new(bytes), variables.device())?;
    let loaded_map: HashMap<_, _> = loaded.into_iter().collect();
    let current = variables.variables();
    if loaded_map.len() != current.len() {
        return Err(TchTrainingError::Invalid(
            "model checkpoint variable count differs",
        ));
    }
    for (name, variable) in current {
        let saved = loaded_map.get(&name).ok_or(TchTrainingError::Invalid(
            "model checkpoint variable is missing",
        ))?;
        if saved.size() != variable.size() || saved.kind() != variable.kind() {
            return Err(TchTrainingError::Invalid(
                "model checkpoint variable shape or type differs",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod model_state_serialization_tests {
    use super::{encode_model_state, validate_model_state};
    use tch::nn::{Init, VarStore};
    use tch::Device;

    fn variables(insert_reverse: bool) -> VarStore {
        let store = VarStore::new(Device::Cpu);
        let root = store.root();
        let add_alpha = || root.var("alpha", &[2, 3], Init::Const(0.25));
        let add_omega = || root.var("omega", &[3], Init::Const(-0.5));
        if insert_reverse {
            let _ = add_omega();
            let _ = add_alpha();
        } else {
            let _ = add_alpha();
            let _ = add_omega();
        }
        store
    }

    #[test]
    fn canonical_encoder_is_order_independent_and_reads_legacy_var_store_streams() {
        let alpha_first = variables(false);
        let omega_first = variables(true);
        let canonical_alpha_first = encode_model_state(&alpha_first).unwrap();
        let canonical_omega_first = encode_model_state(&omega_first).unwrap();
        assert_eq!(canonical_alpha_first, canonical_omega_first);

        let mut legacy_stream = Vec::new();
        alpha_first.save_to_stream(&mut legacy_stream).unwrap();
        validate_model_state(&omega_first, &legacy_stream).unwrap();
        validate_model_state(&alpha_first, &canonical_alpha_first).unwrap();
    }
}
