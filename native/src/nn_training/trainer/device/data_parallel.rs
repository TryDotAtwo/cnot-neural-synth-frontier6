//! Synchronous two-replica CUDA updates with one sampler and one AdamW owner.

use super::{DeviceUpdateLoss, FlatAdamW, NeuralTrainer, StepOutcome, TchTrainingError};
use crate::model::FusionDistanceField;
use tch::{nn::VarStore, Device, Kind, Tensor};

impl NeuralTrainer {
    pub(crate) fn run_device_data_parallel_block<S>(
        &mut self,
        model: &FusionDistanceField,
        variables: &VarStore,
        replica_variables: &mut VarStore,
        attempts: u64,
        batch_size: usize,
        sequential_batch_gradients: bool,
        state: &mut S,
        mut objective: impl FnMut(&mut S, usize, &[usize], usize) -> Result<DeviceUpdateLoss, String>,
        commit_audit: impl FnOnce(&mut S) -> Result<(), TchTrainingError>,
    ) -> Result<StepOutcome, TchTrainingError> {
        if self.poisoned {
            return Err(TchTrainingError::Invalid(
                "trainer must restore a checkpoint after a failed optimizer update",
            ));
        }
        if attempts == 0 || attempts > i64::MAX as u64 || batch_size < 2 {
            return Err(TchTrainingError::Invalid("invalid device update interval"));
        }
        let Device::Cuda(master_index) = variables.device() else {
            return Err(TchTrainingError::Invalid(
                "data-parallel master variables must reside on CUDA",
            ));
        };
        let Device::Cuda(replica_index) = replica_variables.device() else {
            return Err(TchTrainingError::Invalid(
                "data-parallel replica variables must reside on CUDA",
            ));
        };
        if master_index == replica_index || model.device() != variables.device() {
            return Err(TchTrainingError::Invalid(
                "data-parallel model and variable devices must be distinct CUDA devices",
            ));
        }
        let attempted = self
            .progress
            .attempted
            .checked_add(attempts)
            .ok_or(TchTrainingError::Invalid(
                "training attempt counter overflow",
            ))?;
        let mut flat = tch::no_grad(|| FlatAdamW::new(&self.optimizer, model))?;
        let mut replica_pairs = bind_replica_parameters(&self.optimizer, variables, replica_variables)?;
        sync_replica(&mut replica_pairs)?;
        let master_parameters = self
            .optimizer
            .parameter_refs()
            .into_iter()
            .map(Tensor::shallow_clone)
            .collect::<Vec<_>>();
        let replica_parameters = replica_pairs
            .iter()
            .map(|(_, replica)| replica.shallow_clone())
            .collect::<Vec<_>>();
        let parameter_count = master_parameters.len();
        let mut sampler = self.sampler.clone();
        self.poisoned = true;
        let result = (|| {
            for _ in 0..attempts {
                let indices = sampler.next(batch_size)?;
                let mut accumulated: [Vec<Option<Tensor>>; 2] = std::array::from_fn(|_| {
                    (0..parameter_count).map(|_| None).collect()
                });
                let mut losses = Vec::with_capacity(2.max(batch_size));
                let mut votes = Vec::with_capacity(2.max(batch_size));
                let mut groups = Vec::with_capacity(2.max(batch_size));

                if sequential_batch_gradients {
                    for (position, index) in indices.iter().enumerate() {
                        let replica = position % 2;
                        let data = crate::stage_profile::measure("bellman_and_loss", || {
                            objective(state, replica, std::slice::from_ref(index), position)
                        })
                        .map_err(TchTrainingError::Objective)?;
                        accumulate_replica(
                            &flat,
                            replica,
                            1,
                            batch_size,
                            data,
                            &master_parameters,
                            &replica_parameters,
                            &mut accumulated,
                            &mut losses,
                            &mut votes,
                            &mut groups,
                        )?;
                    }
                } else {
                    let split = batch_size.div_ceil(2);
                    let first = &indices[..split];
                    let data = crate::stage_profile::measure("bellman_and_loss", || {
                        objective(state, 0, first, 0)
                    })
                    .map_err(TchTrainingError::Objective)?;
                    accumulate_replica(
                        &flat,
                        0,
                        first.len(),
                        batch_size,
                        data,
                        &master_parameters,
                        &replica_parameters,
                        &mut accumulated,
                        &mut losses,
                        &mut votes,
                        &mut groups,
                    )?;
                    if split < batch_size {
                        let second = &indices[split..];
                        let data = crate::stage_profile::measure("bellman_and_loss", || {
                            objective(state, 1, second, split)
                        })
                        .map_err(TchTrainingError::Objective)?;
                        accumulate_replica(
                            &flat,
                            1,
                            second.len(),
                            batch_size,
                            data,
                            &master_parameters,
                            &replica_parameters,
                            &mut accumulated,
                            &mut losses,
                            &mut votes,
                            &mut groups,
                        )?;
                    }
                }

                let (master_gradient, master_defined) =
                    flatten_gradients(&accumulated[0], &master_parameters);
                let (replica_gradient, replica_defined) =
                    flatten_gradients(&accumulated[1], &replica_parameters);
                let total_gradient = master_gradient + replica_gradient.to_device(variables.device());
                let defined: Vec<_> = master_defined
                    .into_iter()
                    .zip(replica_defined)
                    .map(|(a, b)| a || b)
                    .collect();
                let loss = Tensor::stack(&losses, 0).sum(Kind::Float);
                let update = DeviceUpdateLoss {
                    valid: Tensor::stack(&votes, 0)
                        .all()
                        .logical_and(&loss.isfinite()),
                    group_activity: Tensor::stack(&groups, 0).any_dim(0, false),
                    loss,
                };
                crate::stage_profile::measure("data_parallel_reduce", || {
                    tch::no_grad(|| {
                        flat.update_flat(
                            &mut self.optimizer,
                            &update,
                            &total_gradient,
                            &defined,
                        )
                    })
                })?;
                sync_replica(&mut replica_pairs)?;
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
                sync_replica(&mut replica_pairs)?;
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

fn bind_replica_parameters(
    optimizer: &super::super::super::optimizer::AdamW,
    master: &VarStore,
    replica: &VarStore,
) -> Result<Vec<(Tensor, Tensor)>, TchTrainingError> {
    let names = optimizer.parameter_names();
    let refs = optimizer.parameter_refs();
    if names.len() != refs.len() {
        return Err(TchTrainingError::Invalid("optimizer parameter identity mismatch"));
    }
    let master_values = master.variables();
    let replica_values = replica.variables();
    let mut pairs = Vec::with_capacity(names.len());
    for (name, parameter) in names.iter().zip(refs) {
        let source = master_values
            .get(name)
            .ok_or(TchTrainingError::Invalid("master parameter name is missing"))?;
        let target = replica_values
            .get(name)
            .ok_or(TchTrainingError::Invalid("replica parameter name is missing"))?;
        if source.data_ptr() != parameter.data_ptr()
            || source.kind() != Kind::Float
            || target.kind() != Kind::Float
            || source.size() != target.size()
            || source.device() != master.device()
            || target.device() != replica.device()
            || !target.requires_grad()
        {
            return Err(TchTrainingError::Invalid("replica parameter contract differs"));
        }
        pairs.push((source.shallow_clone(), target.shallow_clone()));
    }
    Ok(pairs)
}

fn sync_replica(pairs: &mut [(Tensor, Tensor)]) -> Result<(), TchTrainingError> {
    tch::no_grad(|| -> Result<(), TchTrainingError> {
        for (source, target) in pairs {
            let source = source.to_device(target.device());
            target.f_copy_(&source)?;
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn accumulate_replica(
    flat: &FlatAdamW,
    replica: usize,
    local_count: usize,
    batch_size: usize,
    data: DeviceUpdateLoss,
    master_parameters: &[Tensor],
    replica_parameters: &[Tensor],
    accumulated: &mut [Vec<Option<Tensor>>; 2],
    losses: &mut Vec<Tensor>,
    votes: &mut Vec<Tensor>,
    groups: &mut Vec<Tensor>,
) -> Result<(), TchTrainingError> {
    if replica > 1 || local_count == 0 || batch_size == 0 || local_count > batch_size {
        return Err(TchTrainingError::Invalid("invalid data-parallel batch partition"));
    }
    if replica == 0 {
        flat.validate_loss(&data)?;
    } else if replica_parameters.is_empty()
        || data.loss.device() != replica_parameters[0].device()
        || data.loss.kind() != Kind::Float
        || data.loss.numel() != 1
        || !data.loss.requires_grad()
        || data.valid.device() != data.loss.device()
        || data.valid.kind() != Kind::Bool
        || data.valid.dim() != 0
        || data.group_activity.device() != data.loss.device()
        || data.group_activity.kind() != Kind::Bool
        || data.group_activity.size() != [flat.group_count()]
    {
        return Err(TchTrainingError::Invalid(
            "replica objective loss, validity or group participation has the wrong contract",
        ));
    }
    let parameters = if replica == 0 {
        master_parameters
    } else {
        replica_parameters
    };
    if parameters.is_empty() || parameters.len() != accumulated[replica].len() {
        return Err(TchTrainingError::Invalid("replica parameter count differs"));
    }
    let scaled = &data.loss * (local_count as f64 / batch_size as f64);
    let refs: Vec<_> = parameters.iter().collect();
    let gradients = crate::stage_profile::measure("backward", || {
        Tensor::f_run_backward(&[&scaled], &refs, false, false)
    })?;
    if gradients.len() != accumulated[replica].len() {
        return Err(TchTrainingError::Invalid(
            "replica autograd gradient count differs from parameter count",
        ));
    }
    tch::no_grad(|| -> Result<(), TchTrainingError> {
        for ((sum, gradient), parameter) in accumulated[replica]
            .iter_mut()
            .zip(gradients)
            .zip(parameters)
        {
            if gradient.defined() {
                if gradient.is_sparse()
                    || gradient.kind() != Kind::Float
                    || gradient.device() != parameter.device()
                    || gradient.size() != parameter.size()
                {
                    return Err(TchTrainingError::Invalid(
                        "replica gradient differs from its parameter contract",
                    ));
                }
                *sum = Some(match sum.take() {
                    Some(previous) => previous + gradient,
                    None => gradient,
                });
            }
        }
        Ok(())
    })?;
    let weight = local_count as f64 / batch_size as f64;
    losses.push(data.loss.detach().to_device(master_parameters[0].device()) * weight);
    votes.push(data.valid.to_device(master_parameters[0].device()));
    groups.push(data.group_activity.to_device(master_parameters[0].device()));
    Ok(())
}

fn flatten_gradients(
    gradients: &[Option<Tensor>],
    parameters: &[Tensor],
) -> (Tensor, Vec<bool>) {
    let mut parts = Vec::with_capacity(parameters.len());
    let mut defined = Vec::with_capacity(parameters.len());
    for (gradient, parameter) in gradients.iter().zip(parameters) {
        match gradient {
            Some(gradient) => {
                parts.push(gradient.reshape([-1]));
                defined.push(true);
            }
            None => {
                parts.push(Tensor::zeros_like(parameter).reshape([-1]));
                defined.push(false);
            }
        }
    }
    (Tensor::cat(&parts, 0), defined)
}
