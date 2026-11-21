//! Read-only evaluation of explicitly selected, pre-frozen dev or test roots.
use super::{
    artifacts,
    evaluation::{load_panel_checkpoint, PanelCheckpoint},
};
use super::super::panel::{frozen_panel_digest, validate_frozen_panel};
use crate::{
    device_search::{search_hamming, search_neural, ActionHead, BeamConfig},
    run_config::manifest::PinnedArtifact,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path, time::Instant};
#[cfg(test)]
mod tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchManifest {
    schema: String,
    training_manifest: PinnedArtifact,
    training_result: PinnedArtifact,
    panel: String,
    panel_indices: Vec<usize>,
    panel_ids: Vec<String>,
    panel_digest: String,
    search: BeamConfig,
    head: ActionHead,
}

pub fn execute_dev_search_manifest(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let search: SearchManifest = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if search.schema != "cnot103_panel_search_v2" {
        return Err("unknown panel search manifest schema".into());
    }
    if !matches!(search.panel.as_str(), "dev" | "test")
        || search.panel_indices.is_empty()
        || search.panel_indices.len() != search.panel_ids.len()
        || search
            .panel_indices
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != search.panel_indices.len()
        || search.panel_ids.iter().any(|id| id.trim().is_empty())
        || search.panel_ids.iter().collect::<BTreeSet<_>>().len() != search.panel_ids.len()
        || frozen_panel_digest(&search.panel, &search.panel_indices, &search.panel_ids)
            != search.panel_digest
    {
        return Err("panel search manifest cohort identity is invalid".into());
    }
    if matches!(search.head, ActionHead::Hamming) {
        return Err("CPU Hamming uses the sealed CPU-only search manifest".into());
    }
    let PanelCheckpoint {
        field,
        receipt,
        audit,
        device,
    } = load_panel_checkpoint(
        path,
        &search.training_manifest,
        &search.training_result,
        &search.panel,
        &search.panel_indices,
    )?;
    validate_frozen_panel(
        &field.dataset,
        &search.panel,
        &search.panel_indices,
        &search.panel_ids,
        &search.panel_digest,
    )?;
    let records = match search.panel.as_str() {
        "dev" => field.dataset.dev.records(),
        "test" => field.dataset.test.records(),
        _ => return Err("panel search manifest panel is invalid".into()),
    };
    let mut rows = Vec::new();
    for (position, &i) in search.panel_indices.iter().enumerate() {
        let record = records
            .get(i)
            .ok_or("panel search index out of range")?;
        if record.lineage_id != search.panel_ids[position] {
            return Err("panel search case identity differs from the sealed dataset".into());
        }
        let root = &record.state;
        let started = Instant::now();
        let out = match search.head {
            ActionHead::Hamming => search_hamming(root, search.search)?,
            head @ (ActionHead::Qhat | ActionHead::Proposal) => search_neural(
                root,
                &field.model,
                &field.plans[&root.dimension()],
                search.search,
                head,
            )?,
        };
        let row_wall_seconds = started.elapsed().as_secs_f64();
        let solved = out.word.is_some();
        let goal = crate::MatrixState::identity(root.dimension(), root.alphabet_id())
            .expect("validated dimension and alphabet");
        let end_packed_rows = if solved {
            Some(goal.packed_rows())
        } else {
            None
        };
        let word = out.word.map(|gates| {
            json!({"control_target":gates.iter().map(|g|[g.control,g.target]).collect::<Vec<_>>(),
                "physical_cost":gates.len(),"replayed":true})
        });
        rows.push(json!({
            "panel_index":i,
            "case_id":record.lineage_id,
            "dimension":root.dimension(),
            "width":root.dimension(),
            "alphabet_id":root.alphabet_id(),
            "start_packed_rows":root.packed_rows(),
            "end_packed_rows":end_packed_rows,
            "status":out.status,
            "word":word,
            "depths":out.depths,
            "expanded":out.expanded,
            "generated":out.generated,
            "value_evaluations":out.value_evaluations,
            "action_evaluations":out.action_evaluations,
            "value_slots":out.value_slots,
            "action_slots":out.action_slots,
            "wall_seconds":row_wall_seconds
        }));
    }
    Ok(json!({
        "schema":"cnot103_panel_search_result_v2",
        "search_manifest_sha256":artifacts::sha256(&raw),
        "training_manifest_sha256":search.training_manifest.sha256,
        "training_result_sha256":search.training_result.sha256,
        "checkpoint":receipt,
        "panel":search.panel,
        "panel_indices":search.panel_indices,
        "panel_ids":search.panel_ids,
        "panel_digest":search.panel_digest,
        "dataset_sha256":field.dataset.sha256,
        "objective_sha256":field.objective.config_sha256(),
        "feature_registry_sha256":field.registry.contract_sha256(),
        "training_exposure_audit":audit,
        "feature_model_inputs":field.selected_count,
        "feature_execution":if matches!(search.head, ActionHead::Hamming) { "not_used_by_hamming" } else { "used_for_model_scoring" },
        "feature_policy_id":field.objective.feature_policy().id().map_err(|e|e.to_string())?,
        "cuda_device":device,
        "search":search.search,
        "head":search.head,
        "execution_backend":if matches!(search.head, ActionHead::Hamming) { "cpu_hamming" } else { "cuda_model" },
        "rows":rows
    }))
}
