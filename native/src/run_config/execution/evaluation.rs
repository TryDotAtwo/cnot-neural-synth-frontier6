//! Admission for read-only evaluation on a frozen dev panel.
use super::{artifacts, audit_json, field::BoundField};
use crate::nn_training::NeuralTrainer;
use crate::run_config::manifest::{parse_run_manifest, PinnedArtifact};
use crate::{CheckpointManifest, CheckpointStore, Sampler, TrainingSplit};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use tch::Tensor;

pub(super) struct DevCheckpoint {
    pub field: BoundField,
    pub receipt: CheckpointManifest,
    pub audit: Value,
    pub device: usize,
}

fn pinned_file(base: &Path, input: &PinnedArtifact) -> Result<(PathBuf, Vec<u8>), String> {
    let p = artifacts::resolve_path(base, &input.path);
    let b = fs::read(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
    if artifacts::sha256(&b) != input.sha256 {
        return Err(format!("pinned SHA-256 differs for {}", p.display()));
    }
    Ok((p, b))
}

pub(super) fn load_dev_checkpoint(
    path: &Path,
    training_manifest: &PinnedArtifact,
    training_result: &PinnedArtifact,
    dev_indices: &[usize],
) -> Result<DevCheckpoint, String> {
    if dev_indices.is_empty()
        || dev_indices.iter().collect::<BTreeSet<_>>().len() != dev_indices.len()
    {
        return Err("evaluation requires explicit distinct dev indices".into());
    }
    let (run_path, run_raw) = pinned_file(path, training_manifest)?;
    let (_, report_raw) = pinned_file(path, training_result)?;
    let report: Value = serde_json::from_slice(&report_raw).map_err(|e| e.to_string())?;
    let manifest = parse_run_manifest(&run_raw)?;
    let paths = artifacts::input_paths(&run_path, &run_raw, &manifest);
    if report["schema"] != "cnot103_training_result_v1"
        || report["training_runtime"] != manifest.objective.runtime_id()
        || report["run_manifest_sha256"] != paths.run_sha256
        || report["dataset_sha256"] != manifest.dataset.sha256
        || report.get("exact_catalog_bundle_sha256")
            != Some(&serde_json::json!(manifest
                .exact_catalogs
                .as_ref()
                .map(|p| p.sha256.as_str())))
        || report["feature_model_inputs"] != 102
    {
        return Err("training result differs from the pinned full-input run".into());
    }
    let receipt: CheckpointManifest =
        serde_json::from_value(report["checkpoint"].clone()).map_err(|e| e.to_string())?;
    if receipt.progress.accepted == 0
        || serde_json::to_value(receipt.progress).map_err(|e| e.to_string())?
            != report["end_progress"]
    {
        return Err("an accepted, progress-bound training checkpoint is required".into());
    }
    let index = manifest.training.cuda_device;
    crate::torch_device::require_cuda_device(index)?;
    let data = artifacts::load(&manifest, &paths)?;
    let mut widths = BTreeSet::new();
    for &i in dev_indices {
        let row = data
            .dataset
            .dev
            .records()
            .get(i)
            .ok_or("dev search index out of range")?;
        if row.split != TrainingSplit::Dev || !data.dataset.forbidden_keys.contains(&row.state) {
            return Err("requested dev root lacks a frozen holdout identity".into());
        }
        widths.insert(row.state.dimension());
    }
    let mut field = BoundField::build(
        data,
        manifest.feature_contexts,
        manifest.feature_policy,
        manifest.model,
        manifest.objective,
        &manifest.training,
        widths,
        None,
    )?;
    field.objective.bind_frozen_dataset(&field.dataset)?;
    if report["objective_sha256"] != field.objective.config_sha256()
        || report["feature_registry_sha256"] != field.registry.contract_sha256()
    {
        return Err("reconstructed objective or registry differs from training".into());
    }
    let contract = field.objective.checkpoint_contract();
    let store = CheckpointStore::new(&paths.checkpoint_dir);
    let snapshot = store
        .load_manifest(&receipt, &contract)
        .map_err(|e| e.to_string())?;
    field
        .objective
        .restore_checkpoint_state(&snapshot.objective_state)?;
    field.objective.verify_prevalidated_device_closure(field.dataset.train.records())?;
    let audit = field.objective.audit();
    if !audit.strict_unseen
        || !matches!(audit.audit_mode, "stored_full_state_identities" | "prevalidated_device_one_step_closure")
        || audit_json(audit) != report["exposure_audit"]
    {
        return Err("dev search requires the matching verified unseen exposure audit".into());
    }
    {
        let sampler = Sampler::new(field.dataset.train.len(), manifest.training.seed)
            .map_err(|e| e.to_string())?;
        let mut trainer = NeuralTrainer::new(&field.variables, sampler, manifest.optimizer)
            .map_err(|e| e.to_string())?;
        trainer
            .restore_checkpoint_payload(&mut field.variables, snapshot, &contract)
            .map_err(|e| e.to_string())?;
    }
    // One admission check before the search; unused nonfinite weights also fail.
    let finite: Vec<_> = field
        .variables
        .variables()
        .values()
        .map(|t| t.isfinite().all())
        .collect();
    if Tensor::stack(&finite, 0).all().int64_value(&[]) == 0 {
        return Err("search checkpoint has nonfinite weights".into());
    }
    Ok(DevCheckpoint {
        field,
        receipt,
        audit: report["exposure_audit"].clone(),
        device: index,
    })
}
