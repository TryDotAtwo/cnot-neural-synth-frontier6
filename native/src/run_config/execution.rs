//! Manifest-driven assembly and execution of one pinned training run.

mod artifacts;
pub(super) mod coverage;
mod evaluation;
mod feature_runtime;
mod field;
pub(super) mod h07;
mod quality;
pub(super) mod search;

use super::manifest::parse_run_manifest;
use crate::nn_training::NeuralTrainer;
use crate::objective::ExposureAudit;
use crate::training::Sampler;
use crate::training_session::{
    FullGraphTrainingSession, TrainingSessionConfig, TrainingSessionResources,
};
use crate::CheckpointStore;
use field::BoundField;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Run real optimizer updates for a bounded timing experiment. This explicitly
/// discards the model and optimizer: use the training CLI for durable work.
pub fn execute_benchmark_manifest(path: &Path) -> Result<Value, String> {
    let profile = crate::stage_profile::Session::start();
    profile.check_chunk_probe()?;
    let started = Instant::now();
    if std::env::var_os("CNOT103_PROFILE_CHUNKS").is_some() {
        return Err("candidate probes are not complete optimizer benchmarks".into());
    }
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let manifest = parse_run_manifest(&raw)?;
    let quality = quality::optional_spec()?;
    if manifest.resume {
        return Err("discard-only benchmark requires resume=false".into());
    }
    let runtime_id = manifest.objective.runtime_id();
    let paths = artifacts::resolve(path, &raw, &manifest)?;
    let devices = manifest.training.effective_cuda_devices();
    for &index in &devices {
        crate::torch_device::require_cuda_device(index)?;
    }
    profile.device(manifest.training.cuda_device);
    let replica_device = devices.get(1).copied();
    let pinned = artifacts::load(&manifest, &paths)?;
    let widths: std::collections::BTreeSet<u32> = pinned.dataset.train.records().iter().map(|r| r.state.dimension()).collect();
    let BoundField { dataset, registry, plans, selected_count, mut objective, model, variables, mut replica } =
        BoundField::build(pinned, manifest.feature_contexts, manifest.feature_policy,
                          manifest.model, manifest.objective, &manifest.training, widths,
                          replica_device)?;
    let (quality_before, quality_before_seconds) = if let Some(spec) = quality.as_ref() {
        let timer = Instant::now();
        let report = quality::evaluate(spec, &dataset, &model, &plans, &variables, selected_count)?;
        for &index in &devices {
            tch::Cuda::synchronize(index as i64);
        }
        (Some(report), timer.elapsed().as_secs_f64())
    } else {
        (None, 0.0)
    };
    let sampler = Sampler::new(dataset.train.len(), manifest.training.seed).map_err(|e| e.to_string())?;
    let mut trainer = NeuralTrainer::new(&variables, sampler, manifest.optimizer).map_err(|e| e.to_string())?;
    objective.bind_frozen_dataset(&dataset)?;
    let (last, evaluation_plans, prepared_seconds, update_seconds) =
        if let Some(replica) = replica.as_mut() {
            let mut runner = crate::objective::FullGraphObjectiveRunner::new(
                &mut objective, &model, &variables, &registry, dataset.train.records(), None,
            )?.into_data_parallel(
                plans,
                &replica.model,
                &mut replica.variables,
                std::mem::take(&mut replica.plans),
            )?;
            for &index in &devices {
                tch::Cuda::synchronize(index as i64);
            }
            let prepared_seconds = (started.elapsed().as_secs_f64() - quality_before_seconds).max(0.0);
            let timer = Instant::now();
            let last = runner.run_interval(
                &mut trainer,
                manifest.training.attempts,
                manifest.training.batch_size,
            ).map_err(|e| format!("benchmark update failed: {e}"))?;
            for &index in &devices {
                tch::Cuda::synchronize(index as i64);
            }
            let update_seconds = timer.elapsed().as_secs_f64();
            let evaluation_plans = if quality.is_some() {
                Some(runner.into_primary_plans())
            } else {
                drop(runner);
                None
            };
            (last, evaluation_plans, prepared_seconds, update_seconds)
        } else {
            let mut runner = crate::objective::FullGraphObjectiveRunner::new(
                &mut objective, &model, &variables, &registry, dataset.train.records(), None,
            )?.into_device(plans)?;
            tch::Cuda::synchronize(manifest.training.cuda_device as i64);
            let prepared_seconds = (started.elapsed().as_secs_f64() - quality_before_seconds).max(0.0);
            let timer = Instant::now();
            let last = runner.run_interval(
                &mut trainer,
                manifest.training.attempts,
                manifest.training.batch_size,
            ).map_err(|e| format!("benchmark update failed: {e}"))?;
            tch::Cuda::synchronize(manifest.training.cuda_device as i64);
            let update_seconds = timer.elapsed().as_secs_f64();
            let evaluation_plans = if quality.is_some() {
                Some(runner.into_plans())
            } else {
                drop(runner);
                None
            };
            (last, evaluation_plans, prepared_seconds, update_seconds)
        };
    let audit = objective.audit();
    if !audit.strict_unseen || trainer.progress().accepted != manifest.training.attempts {
        return Err("benchmark rejected updates or violated the holdout boundary".into());
    }
    let state_digests = match std::env::var("CNOT103_BENCHMARK_STATE_DIGEST") {
        Ok(value) if value == "1" => {
            let (model, optimizer, sampler) = trainer.benchmark_state_digests(&variables)
                .map_err(|e| format!("benchmark state digest failed: {e}"))?;
            Some(json!({"model_sha256": model, "optimizer_sha256": optimizer,
                        "sampler_sha256": sampler}))
        }
        Ok(_) => return Err("CNOT103_BENCHMARK_STATE_DIGEST must be 1 when set".into()),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => return Err("invalid benchmark digest flag".into()),
    };
    let (evaluation, quality_after_seconds) = match (quality.as_ref(), evaluation_plans) {
        (Some(spec), Some(plans)) => {
            let timer = Instant::now();
            let report = quality::evaluate(
                spec, &dataset, &model, &plans, &variables, selected_count,
            )?;
            tch::Cuda::synchronize(manifest.training.cuda_device as i64);
            (Some(report), timer.elapsed().as_secs_f64())
        }
        (None, None) => (None, 0.0),
        _ => unreachable!("evaluation spec and resident plans are paired"),
    };
    let quality_delta = match (&quality_before, &evaluation) {
        (Some(before), Some(after)) => Some(quality::compare(before, after)?),
        (None, None) => None,
        _ => unreachable!("before/after search reports are paired"),
    };
    Ok(json!({
        "schema": "cnot103_discard_only_benchmark_v2",
        "run_manifest_sha256": paths.run_sha256,
        "dataset_sha256": dataset.sha256,
        "objective_sha256": objective.config_sha256(),
        "feature_registry_sha256": registry.contract_sha256(),
        "feature_model_inputs": selected_count,
        "model_parameters": variables.trainable_variables().iter().map(tch::Tensor::numel).sum::<usize>(),
        "training_runtime": runtime_id,
        "train_rows": dataset.train.len(),
        "attempts": manifest.training.attempts,
        "batch_size": manifest.training.batch_size,
        "cuda_devices": devices,
        "progress": trainer.progress(),
        "last_step": {"accepted":last.accepted, "loss":last.loss},
        "exposure_audit": audit_json(audit),
        "setup_seconds": prepared_seconds,
        "update_seconds": update_seconds,
        "quality_before_seconds": quality_before_seconds,
        "quality_after_seconds": quality_after_seconds,
        "total_seconds": started.elapsed().as_secs_f64(),
        "checkpoint_written": false,
        "checkpoint_time_included": false,
        "state_digests": state_digests,
        "quality_before": quality_before,
        "quality_after": evaluation,
        "quality_delta": quality_delta,
    }))
}

pub fn execute_run_manifest(path: &Path) -> Result<Value, String> {
    let profile = crate::stage_profile::Session::start();
    profile.check_chunk_probe()?;
    let raw = fs::read(path)
        .map_err(|e| format!("could not read run manifest {}: {e}", path.display()))?;
    let manifest = parse_run_manifest(&raw)?;
    let runtime_id = manifest.objective.runtime_id();
    if profile.probing_candidates()
        && (manifest.resume || manifest.training.attempts != 1 || manifest.training.batch_size != 1)
    {
        return Err(
            "a candidate chunk probe requires a fresh single-record, single-attempt run".into(),
        );
    }
    let paths = artifacts::resolve(path, &raw, &manifest)?;

    let devices = manifest.training.effective_cuda_devices();
    crate::stage_profile::measure("cuda_init", || -> Result<(), String> {
        for &index in &devices {
            crate::torch_device::require_cuda_device(index)?;
        }
        Ok(())
    })?;
    profile.device(manifest.training.cuda_device);
    let replica_device = devices.get(1).copied();
    let pinned = artifacts::load(&manifest, &paths)?;
    let widths: std::collections::BTreeSet<u32> = pinned
        .dataset
        .train
        .records()
        .iter()
        .map(|r| r.state.dimension())
        .collect();
    let BoundField {
        dataset,
        registry,
        plans,
        selected_count,
        mut objective,
        model,
        mut variables,
        mut replica,
    } = crate::stage_profile::measure("model_and_plans_setup", || {
        BoundField::build(
            pinned,
            manifest.feature_contexts,
            manifest.feature_policy,
            manifest.model,
            manifest.objective,
            &manifest.training,
            widths,
            replica_device,
        )
    })?;
    let sampler =
        Sampler::new(dataset.train.len(), manifest.training.seed).map_err(|e| e.to_string())?;
    let mut trainer =
        NeuralTrainer::new(&variables, sampler, manifest.optimizer).map_err(|e| e.to_string())?;
    let store = CheckpointStore::new(&paths.checkpoint_dir);
    let session_config = TrainingSessionConfig {
        batch_size: manifest.training.batch_size,
        checkpoint_every_attempts: manifest.training.checkpoint_every_attempts,
    };
    let resources = TrainingSessionResources {
        dataset: &dataset,
        objective: &mut objective,
        model: &model,
        variables: &mut variables,
        registry: &registry,
        trainer: &mut trainer,
        store: &store,
        admission: None,
    };
    let report = {
        let mut session = if manifest.resume {
            FullGraphTrainingSession::resume_latest(resources, session_config)
                .map_err(|e| format!("checkpoint resume failed: {e}"))?
        } else {
            FullGraphTrainingSession::new(resources, session_config)
                .map_err(|e| format!("new training session failed: {e}"))?
        };
        if let Some(replica) = replica.as_mut() {
            session
                .run_device_data_parallel_attempts(
                    manifest.training.attempts,
                    plans,
                    &replica.model,
                    &mut replica.variables,
                    std::mem::take(&mut replica.plans),
                )
                .map_err(|e| format!("training session failed: {e}"))?
        } else {
            session
                .run_device_attempts(manifest.training.attempts, plans)
                .map_err(|e| format!("training session failed: {e}"))?
        }
    };
    let audit = objective.audit();
    if !audit.strict_unseen {
        return Err("post-run exposure audit found overlap with forbidden states".into());
    }
    let loss = if report.last_step.loss.is_finite() {
        Value::from(report.last_step.loss)
    } else {
        Value::Null
    };
    Ok(json!({
        "schema": "cnot103_training_result_v1",
        "run_manifest_sha256": paths.run_sha256,
        "dataset_sha256": dataset.sha256,
        "exact_catalog_bundle_sha256": manifest.exact_catalogs.as_ref().map(|p| p.sha256.as_str()),
        "objective_sha256": objective.config_sha256(),
        "feature_registry_sha256": registry.contract_sha256(),
        "feature_model_inputs": selected_count,
        "model_parameters": variables.trainable_variables().iter().map(tch::Tensor::numel).sum::<usize>(),
        "training_runtime": runtime_id,
        "cuda_devices": manifest.training.effective_cuda_devices(),
        "train_rows": dataset.train.len(),
        "dev_rows": dataset.dev.len(),
        "test_rows": dataset.test.len(),
        "start_progress": report.start_progress,
        "end_progress": report.end_progress,
        "last_step": { "accepted": report.last_step.accepted, "loss": loss },
        "exposure_audit": audit_json(audit),
        "checkpoint": report.final_checkpoint,
    }))
}

fn audit_json(a: ExposureAudit) -> Value {
    let mut result = json!({
        "strict_unseen": a.strict_unseen, "audit_mode": a.audit_mode,
        "label_count": a.label_count, "forward_count": a.forward_count, "gradient_count": a.gradient_count,
        "forbidden_count": a.forbidden_count, "label_overlap": a.label_overlap,
        "forward_overlap": a.forward_overlap, "gradient_overlap": a.gradient_overlap,
        "set_sha256": a.set_sha256
    });
    if a.audit_mode.starts_with("prevalidated_device_") {
        result["count_semantics"] = "observed_visits_not_unique_states".into();
    }
    result
}
