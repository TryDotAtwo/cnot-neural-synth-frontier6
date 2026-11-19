//! Read-only, post-forward H07 with an actual frozen training checkpoint.
use super::{
    artifacts,
    evaluation::{load_dev_checkpoint, DevCheckpoint},
};
use crate::{run_config::manifest::PinnedArtifact, view_diagnostic::resident, CheckpointManifest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, path::Path};
#[cfg(test)]
mod tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct H07Manifest {
    schema: String,
    training_manifest: PinnedArtifact,
    training_result: PinnedArtifact,
    dev_indices: Vec<usize>,
}

fn model_binding(run_sha: &str, checkpoint: &CheckpointManifest, registry: &str) -> Value {
    json!({"schema":"cnot103_h07_trained_field_v1","head":"FusionDistanceField/value",
        "training_manifest_sha256":run_sha,"checkpoint_sha256":checkpoint.sha256,
        "checkpoint_contract_sha256":checkpoint.contract_sha256,"feature_registry_sha256":registry})
}

pub fn execute_dev_h07_manifest(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let spec: H07Manifest = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if spec.schema != "cnot103_dev_h07_v1" {
        return Err("unknown dev H07 manifest schema".into());
    }
    let DevCheckpoint {
        field,
        receipt,
        audit,
        device,
    } = load_dev_checkpoint(
        path,
        &spec.training_manifest,
        &spec.training_result,
        &spec.dev_indices,
    )?;
    let binding = model_binding(
        &spec.training_manifest.sha256,
        &receipt,
        &field.registry.contract_sha256(),
    );
    let model_sha = artifacts::sha256(&serde_json::to_vec(&binding).map_err(|e| e.to_string())?);
    let mut rows = Vec::with_capacity(spec.dev_indices.len());
    for i in spec.dev_indices {
        let record = &field.dataset.dev.records()[i];
        let root = &record.state;
        let (scope, diagnostic) = resident::evaluate(
            root,
            field.objective.alphabet_for(root.alphabet_id())?,
            &field.model,
            &field.plans[&root.dimension()],
            &model_sha,
        )
        .map_err(|e| format!("dev[{i}]: {e}"))?;
        rows.push(json!({"dev_index":i,"lineage":record.lineage_id,"problem_scope":scope,"diagnostic":resident::report_json(&diagnostic)}));
    }
    Ok(
        json!({"schema":"cnot103_dev_h07_result_v1","h07_manifest_sha256":artifacts::sha256(&raw),
        "training_result_sha256":spec.training_result.sha256,"checkpoint":receipt,"model_binding":binding,"dataset_sha256":field.dataset.sha256,
        "training_exposure_audit":audit,"panel":"dev","feature_model_inputs":field.selected_count,"cuda_device":device,"rows":rows}),
    )
}
