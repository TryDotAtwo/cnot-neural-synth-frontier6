//! Read-only evaluation of explicitly selected, pre-frozen dev roots. The only
//! arguments crossing into the beam are a matrix, trained model and work policy.
use super::{
    artifacts,
    evaluation::{load_dev_checkpoint, DevCheckpoint},
};
use crate::{
    device_search::{search_neural, ActionHead, BeamConfig},
    run_config::manifest::PinnedArtifact,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, path::Path};
#[cfg(test)]
mod tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchManifest {
    schema: String,
    training_manifest: PinnedArtifact,
    training_result: PinnedArtifact,
    dev_indices: Vec<usize>,
    search: BeamConfig,
    head: ActionHead,
}

pub fn execute_dev_search_manifest(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let search: SearchManifest = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if search.schema != "cnot103_dev_search_v1" {
        return Err("unknown dev search manifest schema".into());
    }
    let DevCheckpoint {
        field,
        receipt,
        audit,
        device,
    } = load_dev_checkpoint(
        path,
        &search.training_manifest,
        &search.training_result,
        &search.dev_indices,
    )?;
    let mut rows = Vec::new();
    for &i in &search.dev_indices {
        let record = &field.dataset.dev.records()[i];
        let root = &record.state;
        let out = search_neural(
            root,
            &field.model,
            &field.plans[&root.dimension()],
            search.search,
            search.head,
        )?;
        let word = out.word.map(|gates| {
            let goal = crate::MatrixState::identity(root.dimension(), root.alphabet_id())
                .expect("validated dimension and alphabet");
            json!({"control_target":gates.iter().map(|g|[g.control,g.target]).collect::<Vec<_>>(),
                "end_packed_rows":goal.packed_rows(),"physical_cost":gates.len(),"replayed":true})
        });
        rows.push(json!({"dev_index":i,"lineage":record.lineage_id,"dimension":root.dimension(),"alphabet_id":root.alphabet_id(),
            "start_packed_rows":root.packed_rows(),"status":out.status,"word":word,"depths":out.depths,
            "expanded":out.expanded,"generated":out.generated,"value_evaluations":out.value_evaluations,"action_evaluations":out.action_evaluations,
            "value_slots":out.value_slots,"action_slots":out.action_slots}));
    }
    Ok(
        json!({"schema":"cnot103_dev_search_result_v1","search_manifest_sha256":artifacts::sha256(&raw),
        "training_manifest_sha256":search.training_manifest.sha256,"training_result_sha256":search.training_result.sha256,"checkpoint":receipt,
        "panel":"dev","dataset_sha256":field.dataset.sha256,"objective_sha256":field.objective.config_sha256(),"feature_registry_sha256":field.registry.contract_sha256(),
        "training_exposure_audit":audit,"feature_model_inputs":field.selected_count,"feature_policy_id":field.objective.feature_policy().id().map_err(|e|e.to_string())?,"cuda_device":device,
        "search":search.search,"head":search.head,"rows":rows}),
    )
}
