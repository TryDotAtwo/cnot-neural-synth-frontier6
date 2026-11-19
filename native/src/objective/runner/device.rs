//! CUDA execution of a runner whose model, registry, admission and ordered
//! records are already bound to the checkpoint contract.
use super::FullGraphObjectiveRunner;
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::nn_training::{NeuralTrainer, StepOutcome, TchTrainingError};
use crate::objective::training::device::{BatchAudit, DeviceTrainingData};
use crate::objective::{ExposureLedger, FullGraphObjective};
use crate::training_data::TrainingRecord;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use tch::{nn::VarStore, Device, Tensor};

pub struct DeviceFullGraphRunner<'a> {
    pub(crate) source: FullGraphObjectiveRunner<'a>,
    data: DeviceTrainingData,
}

impl<'a> FullGraphObjectiveRunner<'a> {
    pub fn into_device(
        self,
        plans: HashMap<u32, DeviceFeaturePlan>,
    ) -> Result<DeviceFullGraphRunner<'a>, String> {
        let Device::Cuda(index) = self.model.device() else {
            return Err("device full-graph runner requires CUDA".into());
        };
        if self.objective.policy.evaluations.is_some() {
            return Err(
                "Bellman selection requires frozen unlimited deterministic evaluations".into(),
            );
        }
        if self.objective.config.prevalidated_device_closure {
            self.objective.closure_verified = false;
        }
        let data = DeviceTrainingData::prepare(
            self.objective,
            self.records,
            self.admission,
            plans,
            index,
        )?;
        // `prepare` has checked every record and every primitive successor;
        // the runner is bound to exactly those ordered records and alphabets.
        if self.objective.config.prevalidated_device_closure {
            self.objective.closure_verified = true;
        }
        Ok(DeviceFullGraphRunner { source: self, data })
    }

    pub fn into_data_parallel(
        self,
        primary_plans: HashMap<u32, DeviceFeaturePlan>,
        replica_model: &'a crate::model::FusionDistanceField,
        replica_variables: &'a mut VarStore,
        replica_plans: HashMap<u32, DeviceFeaturePlan>,
    ) -> Result<DeviceDataParallelRunner<'a>, String> {
        let Device::Cuda(primary_index) = self.model.device() else {
            return Err("data-parallel runner requires a CUDA master model".into());
        };
        let Device::Cuda(replica_index) = replica_model.device() else {
            return Err("data-parallel replica model requires CUDA".into());
        };
        if primary_index == replica_index || replica_variables.device() != replica_model.device() {
            return Err("data-parallel models must use distinct CUDA variable stores".into());
        }
        if self.objective.policy.evaluations.is_some() {
            return Err(
                "Bellman selection requires frozen unlimited deterministic evaluations".into(),
            );
        }
        if self.objective.config.prevalidated_device_closure {
            self.objective.closure_verified = false;
        }
        let primary = DeviceTrainingData::prepare(
            self.objective,
            self.records,
            self.admission,
            primary_plans,
            primary_index,
        )?;
        let replica = DeviceTrainingData::prepare(
            self.objective,
            self.records,
            self.admission,
            replica_plans,
            replica_index,
        )?;
        if self.objective.config.prevalidated_device_closure {
            self.objective.closure_verified = true;
        }
        Ok(DeviceDataParallelRunner {
            source: self,
            primary,
            replica_model,
            replica_variables,
            replica,
        })
    }
}

pub struct DeviceDataParallelRunner<'a> {
    pub(crate) source: FullGraphObjectiveRunner<'a>,
    primary: DeviceTrainingData,
    replica_model: &'a crate::model::FusionDistanceField,
    replica_variables: &'a mut VarStore,
    replica: DeviceTrainingData,
}

impl DeviceFullGraphRunner<'_> {
    /// Finish a bounded training run and recover its feature plans for an
    /// in-memory evaluation of the same model.
    pub fn into_plans(self) -> HashMap<u32, DeviceFeaturePlan> {
        self.data.into_plans()
    }

    pub fn run_interval(
        &mut self,
        trainer: &mut NeuralTrainer,
        attempts: u64,
        batch_size: usize,
    ) -> Result<StepOutcome, TchTrainingError> {
        let model = self.source.model;
        let records = self.source.records;
        let data = &self.data;
        let mut attempt = trainer.progress().attempted;
        if self.source.objective.config.sequential_batch_gradients {
            return trainer.run_device_microbatch_block(
                model,
                attempts,
                batch_size,
                &mut Vec::new(),
                |journal, indices, position| {
                    let (loss, audit) = data.compute_at_position(model, indices, attempt, position)?;
                    if position + 1 == batch_size {
                        attempt = attempt.wrapping_add(1);
                    }
                    journal.push(audit);
                    Ok(loss)
                },
                |journal| commit_exposures(self.source.objective, records, journal, batch_size),
            );
        }
        trainer.run_device_block(
            model,
            attempts,
            batch_size,
            &mut Vec::new(),
            |journal, indices| {
                let (loss, audit) = data.compute(model, indices, attempt)?;
                attempt = attempt.wrapping_add(1);
                journal.push(audit);
                Ok(loss)
            },
            |journal| commit_exposures(self.source.objective, records, journal, 1),
        )
    }
}

impl DeviceDataParallelRunner<'_> {
    /// Recover the master-device feature plans after a discard-only run.
    pub fn into_primary_plans(self) -> HashMap<u32, DeviceFeaturePlan> {
        self.primary.into_plans()
    }

    pub fn run_interval(
        &mut self,
        trainer: &mut NeuralTrainer,
        attempts: u64,
        batch_size: usize,
    ) -> Result<StepOutcome, TchTrainingError> {
        let DeviceDataParallelRunner {
            source,
            primary,
            replica_model,
            replica_variables,
            replica,
        } = self;
        let records = source.records;
        let sequential = source.objective.config.sequential_batch_gradients;
        let audit_entries = if sequential { batch_size } else { batch_size.min(2) };
        let mut attempt = trainer.progress().attempted;
        trainer.run_device_data_parallel_block(
            source.model,
            source.variables,
            replica_variables,
            attempts,
            batch_size,
            sequential,
            &mut Vec::new(),
            |journal, replica_index, indices, position| {
                let (loss, audit) = if replica_index == 0 {
                    primary.compute_at_position(source.model, indices, attempt, position)?
                } else {
                    replica.compute_at_position(
                        replica_model,
                        indices,
                        attempt,
                        position,
                    )?
                };
                journal.push(audit);
                if position + indices.len() == batch_size {
                    attempt = attempt.wrapping_add(1);
                }
                Ok(loss)
            },
            |journal| {
                commit_exposures(source.objective, records, journal, audit_entries)
            },
        )
    }
}

/// Materialize the source exposure ledger only at the checkpoint boundary.
/// Selection indices stay resident until the optimizer's global validity vote
/// passes. Reconstructing audit identities never supplies an input to the model.
fn commit_exposures(
    objective: &mut FullGraphObjective,
    records: &[TrainingRecord],
    journal: &[BatchAudit],
    last_batch_entries: usize,
) -> Result<(), TchTrainingError> {
    if last_batch_entries == 0 || journal.len() < last_batch_entries {
        return Err(TchTrainingError::Invalid("missing device audit interval"));
    }
    let mut diagnostics = crate::objective::ObjectiveDiagnostics::default();
    for entry in journal.iter().rev().take(last_batch_entries) {
        let source = &entry.diagnostics;
        diagnostics.examples = diagnostics.examples.checked_add(source.examples)
            .ok_or(TchTrainingError::Invalid("device audit diagnostics overflow"))?;
        diagnostics.primitive_candidates = diagnostics.primitive_candidates.checked_add(source.primitive_candidates)
            .ok_or(TchTrainingError::Invalid("device audit diagnostics overflow"))?;
        diagnostics.exact_examples = diagnostics.exact_examples.checked_add(source.exact_examples)
            .ok_or(TchTrainingError::Invalid("device audit diagnostics overflow"))?;
        diagnostics.teacher_examples = diagnostics.teacher_examples.checked_add(source.teacher_examples)
            .ok_or(TchTrainingError::Invalid("device audit diagnostics overflow"))?;
    }
    let chunks: Vec<_> = journal
        .iter()
        .map(|b| b.selected.to_device(Device::Cpu))
        .collect();
    let selected = Vec::<i64>::try_from(Tensor::cat(&chunks, 0))?;
    let roots: Vec<_> = journal
        .iter()
        .flat_map(|b| b.records.iter().copied())
        .collect();
    if selected.len() != roots.len() {
        return Err(TchTrainingError::Invalid("device audit count mismatch"));
    }
    let sampled = objective.config.sampled_neighbor_count.is_some();
    let mut lengths = Vec::new();
    let mut tensors = Vec::new();
    for entry in journal {
        match (&entry.sampled_candidates, sampled) {
            (Some(rows), true) if rows.len() == entry.records.len() => {
                for row in rows {
                    lengths.push(row.numel());
                    tensors.push(row.to_device(Device::Cpu));
                }
            }
            (None, false) => {}
            _ => return Err(TchTrainingError::Invalid("sampled action audit count mismatch")),
        }
    }
    let candidates = if sampled {
        Vec::<i64>::try_from(Tensor::cat(&tensors, 0))?
    } else {
        Vec::new()
    };
    let mut candidate_offset = 0usize;
    let mut delta = ExposureLedger::default();
    let mut visited: BTreeMap<usize, (bool, BTreeSet<usize>, BTreeSet<usize>)> = BTreeMap::new();
    for (position, (row, selection)) in roots.into_iter().zip(selected).enumerate() {
        let record = &records[row];
        let alphabet = &objective.alphabets[record.state.alphabet_id()];
        let terminal = record.state.is_identity();
        let row_candidates = if sampled {
            let end = candidate_offset.checked_add(lengths[position])
                .ok_or(TchTrainingError::Invalid("sampled action audit overflow"))?;
            let slice = candidates.get(candidate_offset..end)
                .ok_or(TchTrainingError::Invalid("sampled action audit extent"))?;
            candidate_offset = end;
            let expected = if !terminal || objective.config.qhat_weight > 0.0 {
                alphabet.gates().len().min(objective.config.sampled_neighbor_count.unwrap())
            } else { 0 };
            if slice.len() != expected
                || slice.iter().any(|&i| i < 0 || i as usize >= alphabet.gates().len())
                || slice.windows(2).any(|pair| pair[0] >= pair[1])
                || selection >= 0 && slice.binary_search(&selection).is_err()
            {
                return Err(TchTrainingError::Invalid("invalid sampled action audit"));
            }
            slice
        } else { &[] };
        if selection < -2
            || selection == -1 && !terminal
            || selection >= 0 && (terminal || selection as usize >= alphabet.gates().len())
        {
            return Err(TchTrainingError::Invalid(
                "invalid device gradient exposure index",
            ));
        }
        if objective.config.compact_exposures {
            let forwards = 1 + if sampled {
                row_candidates.len() as u64
            } else if !terminal || objective.config.qhat_weight > 0.0 {
                alphabet.gates().len() as u64
            } else {
                0
            };
            for (counter, increment) in delta.compact_counts.iter_mut().zip([
                u64::from(objective.config.supervision_weight > 0.0),
                forwards,
                u64::from(selection != -2) + u64::from(selection >= 0),
            ]) {
                *counter = counter.saturating_add(increment);
            }
        } else {
            let (gradient, children, forwards) = visited.entry(row).or_default();
            *gradient |= selection != -2;
            if selection >= 0 {
                children.insert(selection as usize);
            }
            forwards.extend(row_candidates.iter().map(|&i| i as usize));
        }
    }
    if sampled && candidate_offset != candidates.len() {
        return Err(TchTrainingError::Invalid("sampled action audit trailing entries"));
    }
    for (row, (gradient, children, forwards)) in visited {
        let state = &records[row].state;
        if objective.config.supervision_weight > 0.0 {
            delta.observe(false, 0, state);
        }
        delta.observe(false, 1, state);
        if gradient {
            delta.observe(false, 2, state);
        }
        if !state.is_identity() || objective.config.qhat_weight > 0.0 {
            let alphabet = &objective.alphabets[state.alphabet_id()];
            let indices: Box<dyn Iterator<Item = usize>> = if sampled {
                Box::new(forwards.into_iter())
            } else {
                Box::new(0..alphabet.gates().len())
            };
            for i in indices {
                let gate = alphabet.gates()[i];
                let mut child = state.clone();
                child
                    .apply_cnot(gate)
                    .map_err(|e| TchTrainingError::Objective(e.to_string()))?;
                delta.observe(false, 1, &child);
                if children.contains(&i) {
                    delta.observe(false, 2, &child);
                }
            }
        }
    }
    // All indices and endpoint reconstruction have succeeded before mutation.
    objective
        .exposures
        .seen_as_label
        .extend(delta.seen_as_label);
    objective
        .exposures
        .seen_by_forward
        .extend(delta.seen_by_forward);
    objective
        .exposures
        .seen_by_gradient
        .extend(delta.seen_by_gradient);
    for (total, increment) in objective
        .exposures
        .compact_counts
        .iter_mut()
        .zip(delta.compact_counts)
    {
        *total = total.saturating_add(increment);
    }
    objective.last_diagnostics = diagnostics;
    Ok(())
}
