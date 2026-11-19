//! Flat FP32 AdamW state for the native structural model. Only the interval
//! boundary exports counters. Proposals for all parameters vote before commit.
use super::state::AdamW;
use crate::model::FusionDistanceField;
use crate::nn_training::{DeviceUpdateLoss, StepOutcome, TchTrainingError};
use tch::{Device, Kind, Tensor};

pub(in crate::nn_training) struct FlatAdamW {
    original: Tensor,
    weights: Tensor,
    first: Tensor,
    second: Tensor,
    zero_gradient: Tensor,
    owners: Tensor,
    group_slots: Tensor,
    groups: i64,
    offsets: Vec<(i64, i64)>,
    base_steps: Tensor,
    increments: Tensor,
    accepted: Tensor,
    invalid_objective: Tensor,
    invalid_update: Tensor,
    last_accepted: Tensor,
    last_loss: Tensor,
}

impl FlatAdamW {
    pub(in crate::nn_training) fn new(
        optimizer: &AdamW,
        model: &FusionDistanceField,
    ) -> Result<Self, TchTrainingError> {
        let device = model.device();
        if !matches!(device, Device::Cuda(_)) {
            return Err(TchTrainingError::Invalid(
                "device AdamW requires a CUDA model",
            ));
        }
        let bindings = model.group_parameter_slots();
        let groups = bindings.values().max().map_or(0, |&v| v + 1);
        let mut offset = 0i64;
        let mut offsets = Vec::new();
        let mut slots = Vec::new();
        let mut weights = Vec::new();
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut base_steps = Vec::new();
        if optimizer.parameters.len() > i32::MAX as usize {
            return Err(TchTrainingError::Invalid(
                "optimizer parameter count exceeds int32 owner range",
            ));
        }
        for s in &optimizer.parameters {
            if s.parameter.kind() != Kind::Float || s.parameter.device() != device {
                return Err(TchTrainingError::Invalid(
                    "native field AdamW requires resident FP32 parameters on the model device",
                ));
            }
            let size = i64::try_from(s.parameter.numel())
                .map_err(|_| TchTrainingError::Invalid("optimizer extent overflow"))?;
            offsets.push((offset, size));
            offset = offset
                .checked_add(size)
                .ok_or(TchTrainingError::Invalid("optimizer extent overflow"))?;
            slots.push(
                bindings
                    .get(&(s.parameter.data_ptr() as usize))
                    .copied()
                    .unwrap_or(groups),
            );
            weights.push(s.parameter.detach().reshape([-1]));
            first.push(s.first_moment.reshape([-1]));
            second.push(s.second_moment.reshape([-1]));
            base_steps.push(s.steps as i64);
        }
        // Every element needs only the number of its parameter tensor, not a
        // 64-bit offset. Fill one contiguous map to avoid retaining and then
        // concatenating a second parameter-sized array at interval startup.
        let owners = Tensor::empty([offset], (Kind::Int, device));
        for (i, &(start, size)) in offsets.iter().enumerate() {
            let _ = owners.narrow(0, start, size).fill_(i as i64);
        }
        let original = Tensor::cat(&weights, 0);
        let zeros = Tensor::zeros([], (Kind::Bool, device));
        Ok(Self {
            weights: original.shallow_clone(),
            zero_gradient: Tensor::zeros_like(&original),
            original,
            first: Tensor::cat(&first, 0),
            second: Tensor::cat(&second, 0),
            owners,
            group_slots: Tensor::from_slice(&slots).to_device(device),
            groups,
            offsets,
            base_steps: Tensor::from_slice(&base_steps).to_device(device),
            increments: Tensor::zeros([slots.len() as i64], (Kind::Int64, device)),
            accepted: Tensor::zeros([], (Kind::Int64, device)),
            invalid_objective: zeros.shallow_clone(),
            invalid_update: zeros.shallow_clone(),
            last_accepted: zeros,
            last_loss: Tensor::zeros([], (Kind::Float, device)),
        })
    }

    pub(in crate::nn_training) fn validate_loss(
        &self,
        data: &DeviceUpdateLoss,
    ) -> Result<(), TchTrainingError> {
        let device = self.weights.device();
        if data.loss.device() != device
            || data.loss.kind() != Kind::Float
            || data.loss.numel() != 1
            || !data.loss.requires_grad()
            || data.valid.device() != device
            || data.valid.kind() != Kind::Bool
            || data.valid.dim() != 0
            || data.group_activity.device() != device
            || data.group_activity.kind() != Kind::Bool
            || data.group_activity.size() != [self.groups]
        {
            return Err(TchTrainingError::Invalid(
                "device objective loss, validity or group participation has the wrong contract",
            ));
        }
        Ok(())
    }

    pub(in crate::nn_training) fn group_count(&self) -> i64 {
        self.groups
    }

    pub(in crate::nn_training) fn update(
        &mut self,
        optimizer: &mut AdamW,
        data: &DeviceUpdateLoss,
        gradients: &[Tensor],
    ) -> Result<(), TchTrainingError> {
        if gradients.len() != self.offsets.len() {
            return Err(TchTrainingError::Invalid(
                "autograd gradient count differs from parameter count",
            ));
        }
        let device = self.weights.device();
        let mut defined = Vec::with_capacity(gradients.len());
        let mut parts = Vec::with_capacity(gradients.len());
        for (gradient, &(offset, size)) in gradients.iter().zip(&self.offsets) {
            if gradient.defined() {
                if gradient.is_sparse() {
                    return Err(TchTrainingError::Invalid(
                        "AdamW does not support sparse gradients",
                    ));
                }
                defined.push(true);
                parts.push(gradient.reshape([-1]));
            } else {
                defined.push(false);
                parts.push(self.zero_gradient.narrow(0, offset, size));
            }
        }
        let gradient = Tensor::cat(&parts, 0);
        self.update_flat(optimizer, data, &gradient, &defined)
    }

    pub(in crate::nn_training) fn update_flat(
        &mut self,
        optimizer: &mut AdamW,
        data: &DeviceUpdateLoss,
        gradient: &Tensor,
        defined: &[bool],
    ) -> Result<(), TchTrainingError> {
        let device = self.weights.device();
        if defined.len() != self.offsets.len()
            || gradient.device() != device
            || gradient.kind() != Kind::Float
            || gradient.size() != self.weights.size()
            || data.loss.device() != device
            || data.loss.kind() != Kind::Float
            || data.loss.numel() != 1
            || data.valid.device() != device
            || data.valid.kind() != Kind::Bool
            || data.valid.dim() != 0
            || data.group_activity.device() != device
            || data.group_activity.kind() != Kind::Bool
            || data.group_activity.size() != [self.groups]
        {
            return Err(TchTrainingError::Invalid(
                "flat optimizer update has a mismatched device or tensor contract",
            ));
        }
        let yes = Tensor::ones([], (Kind::Bool, device));
        let defined_values: Vec<i64> = defined.iter().map(|&value| i64::from(value)).collect();
        let defined = Tensor::from_slice(&defined_values)
            .to_device(device)
            .to_kind(Kind::Bool);
        let group_flags = Tensor::cat(&[&data.group_activity, &yes.view([1])], 0);
        let active = group_flags
            .index_select(0, &self.group_slots)
            .logical_and(&defined);
        let element = active.index_select(0, &self.owners);
        let inactive = element.logical_not();
        let any = active.any();
        let finite_gradient = gradient.isfinite().logical_or(&inactive).all();
        let finite_loss = data.loss.isfinite().view([]);
        let alive = self
            .invalid_objective
            .logical_or(&self.invalid_update)
            .logical_not();
        let bad_objective = data
            .valid
            .logical_not()
            .logical_or(&finite_loss.logical_and(&any.logical_not()));
        self.invalid_objective = self
            .invalid_objective
            .logical_or(&alive.logical_and(&bad_objective));
        let eligible = alive
            .logical_and(&data.valid)
            .logical_and(&finite_loss)
            .logical_and(&any)
            .logical_and(&finite_gradient);

        let cfg = optimizer.config;
        let first = &self.first * cfg.beta1 + &gradient * (1.0 - cfg.beta1);
        let second = &self.second * cfg.beta2 + gradient.pow_tensor_scalar(2.0) * (1.0 - cfg.beta2);
        // Checkpoint counters are unsigned. Convert their two 32-bit limbs to
        // FP64 only after adding the exact device-local update count.
        let next = &self.base_steps + &self.increments + 1;
        let step = next
            .bitwise_right_shift_tensor_scalar(32i64)
            .bitwise_and(0xffff_ffffi64)
            .to_kind(Kind::Double)
            * 4294967296.0
            + next.bitwise_and(0xffff_ffffi64).to_kind(Kind::Double);
        let c1 = 1.0f64 - Tensor::pow_scalar(cfg.beta1, &step);
        let c2 = 1.0f64 - Tensor::pow_scalar(cfg.beta2, &step);
        let scale = (cfg.learning_rate / c1)
            .to_kind(Kind::Float)
            .index_select(0, &self.owners);
        let divisor = c2.sqrt().to_kind(Kind::Float).index_select(0, &self.owners);
        let update = &first * scale / (second.sqrt() / divisor + cfg.epsilon);
        let weights = &self.weights * (1.0 - cfg.learning_rate * cfg.weight_decay) - &update;
        let finite_update = first
            .isfinite()
            .logical_and(&second.isfinite())
            .logical_and(&update.isfinite())
            .logical_and(&weights.isfinite())
            .logical_or(&inactive)
            .all();
        self.invalid_update = self
            .invalid_update
            .logical_or(&eligible.logical_and(&finite_update.logical_not()));
        let accept = eligible.logical_and(&finite_update);
        let write = element.logical_and(&accept);
        self.first = first.where_self(&write, &self.first);
        self.second = second.where_self(&write, &self.second);
        self.weights = weights.where_self(&write, &self.weights);
        self.increments = &self.increments + active.logical_and(&accept).to_kind(Kind::Int64);
        self.accepted = &self.accepted + accept.to_kind(Kind::Int64);
        self.last_accepted = accept;
        self.last_loss = data.loss.detach().view([]).copy();
        self.copy_weights(optimizer, &self.weights)
    }

    fn copy_weights(&self, optimizer: &mut AdamW, flat: &Tensor) -> Result<(), TchTrainingError> {
        for (s, &(offset, size)) in optimizer.parameters.iter_mut().zip(&self.offsets) {
            s.parameter.f_copy_(
                &flat
                    .narrow(0, offset, size)
                    .view(s.parameter.size().as_slice()),
            )?;
        }
        Ok(())
    }
    pub(in crate::nn_training) fn rollback(
        &self,
        optimizer: &mut AdamW,
    ) -> Result<(), TchTrainingError> {
        self.copy_weights(optimizer, &self.original)
    }
    /// Called only at an explicit checkpoint or caller-requested report boundary.
    pub(in crate::nn_training) fn finish(
        &self,
        optimizer: &mut AdamW,
        commit_audit: impl FnOnce() -> Result<(), TchTrainingError>,
    ) -> Result<(u64, StepOutcome), TchTrainingError> {
        let flags = Tensor::stack(
            &[
                &self.invalid_objective,
                &self.invalid_update,
                &self.last_accepted,
            ],
            0,
        )
        .to_kind(Kind::Int64);
        let counts = Tensor::cat(&[&flags, &self.accepted.view([1]), &self.increments], 0)
            .to_device(Device::Cpu);
        let counts = Vec::<i64>::try_from(&counts)?;
        if counts[1] != 0 {
            return Err(TchTrainingError::Invalid(super::NONFINITE_CANDIDATE));
        }
        if counts[0] != 0 {
            return Err(TchTrainingError::Objective(
                "device objective or gradient participation check failed".into(),
            ));
        }
        let accepted = counts[3] as u64;
        let last = StepOutcome {
            accepted: counts[2] != 0,
            loss: self.last_loss.double_value(&[]),
        };
        let steps = optimizer
            .steps
            .checked_add(accepted)
            .ok_or(TchTrainingError::Invalid("optimizer step counter overflow"))?;
        let next = optimizer
            .parameters
            .iter()
            .zip(&counts[4..])
            .map(|(state, &delta)| {
                state
                    .steps
                    .checked_add(delta as u64)
                    .ok_or(TchTrainingError::Invalid("parameter step counter overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        commit_audit()?;
        for ((s, &(offset, size)), step) in
            optimizer.parameters.iter_mut().zip(&self.offsets).zip(next)
        {
            let shape = s.parameter.size();
            s.first_moment = self.first.narrow(0, offset, size).view(shape.as_slice());
            s.second_moment = self.second.narrow(0, offset, size).view(shape.as_slice());
            s.steps = step;
        }
        optimizer.steps = steps;
        Ok((accepted, last))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use crate::nn_training::AdamWConfig;
    use crate::{FeatureId, FeaturePolicy, FeatureRegistry};
    use tch::nn::VarStore;

    #[test]
    fn owner_map_fits_four_bytes_per_parameter_and_maps_each_tensor() {
        crate::torch_device::require_cuda_device(0).unwrap();
        let device = Device::Cuda(0);
        let variables = VarStore::new(device);
        let registry = FeatureRegistry::new();
        let mut policy = FeaturePolicy::all_model_inputs_augmentation_only("owner-map-test");
        policy.enabled_ids = vec![FeatureId::new(b'A', 1).unwrap()];
        let model = FusionDistanceField::new_for_policy(
            &variables.root(),
            &registry,
            &policy,
            device,
            ModelConfig {
                wire_dim: 8,
                pair_dim: 4,
                structural_blocks: 1,
                attention_heads: 2,
                feedforward_dim: 16,
                fusion_blocks: 1,
                basis_proposal_hidden_dim: 8,
            },
        )
        .unwrap();
        let optimizer = AdamW::new(&variables, AdamWConfig::default()).unwrap();
        let flat = FlatAdamW::new(&optimizer, &model).unwrap();
        let actual_bytes = flat.owners.numel() * flat.owners.kind().elt_size_in_bytes();
        let parameter_count: usize = optimizer
            .parameters
            .iter()
            .map(|p| p.parameter.numel())
            .sum();
        assert!(
            actual_bytes <= parameter_count * 4,
            "owner map uses {actual_bytes} bytes for {parameter_count} scalar parameters"
        );
        let actual =
            Vec::<i64>::try_from(&flat.owners.to_kind(Kind::Int64).to_device(Device::Cpu)).unwrap();
        let expected: Vec<i64> = optimizer
            .parameters
            .iter()
            .enumerate()
            .flat_map(|(i, p)| std::iter::repeat_n(i as i64, p.parameter.numel()))
            .collect();
        assert_eq!(actual, expected);
    }
}
