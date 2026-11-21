//! CPU-only deterministic Hamming search over a pinned sealed panel.
use super::panel::{
    frozen_panel_digest, load_pinned_panel, validate_panel_root_pins, PanelRootPin,
};
use crate::device_search::{search_hamming, BeamConfig};
use crate::frozen_data::FrozenDataset;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SCHEMA: &str = "cnot103_hamming_search_v1";
const RUNTIME_CONTRACT: &str = "native_device_search_v13";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetPin {
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HammingSearchManifest {
    schema: String,
    source_revision: String,
    dataset: DatasetPin,
    panel: String,
    panel_indices: Vec<usize>,
    panel_ids: Vec<String>,
    panel_digest: String,
    panel_roots: Vec<PanelRootPin>,
    runtime_contract: String,
    search: BeamConfig,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn resolve_path(manifest_path: &Path, configured: &Path) -> PathBuf {
    if configured.is_absolute() {
        configured.to_owned()
    } else {
        manifest_path.parent().unwrap_or_else(|| Path::new(".")).join(configured)
    }
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn execute_hamming_search_manifest(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path).map_err(|_| "Hamming manifest is unavailable".to_owned())?;
    let manifest: HammingSearchManifest =
        serde_json::from_slice(&raw).map_err(|_| "Hamming manifest is invalid".to_owned())?;
    if manifest.schema != SCHEMA
        || manifest.runtime_contract != RUNTIME_CONTRACT
        || !valid_revision(&manifest.source_revision)
        || frozen_panel_digest(
            &manifest.panel,
            &manifest.panel_indices,
            &manifest.panel_ids,
        ) != manifest.panel_digest
    {
        return Err("Hamming manifest identity or runtime contract is invalid".into());
    }
    let dataset_path = resolve_path(path, &manifest.dataset.path);
    let dataset = load_pinned_panel(
        &dataset_path,
        &manifest.dataset.sha256,
        &manifest.panel,
        &manifest.panel_indices,
        &manifest.panel_ids,
        &manifest.panel_digest,
    )?;
    let panel = match manifest.panel.as_str() {
        "dev" => &dataset.dev,
        "test" => &dataset.test,
        _ => return Err("Hamming panel must be dev or test".into()),
    };
    validate_panel_root_pins(
        panel,
        &manifest.panel_indices,
        &manifest.panel_ids,
        &manifest.panel_roots,
    )?;

    let mut rows = Vec::with_capacity(manifest.panel_indices.len());
    for (position, (&index, case_id)) in manifest
        .panel_indices
        .iter()
        .zip(&manifest.panel_ids)
        .enumerate()
    {
        let record = panel.records().get(index).ok_or("Hamming panel index is out of range")?;
        if record.lineage_id != *case_id {
            return Err("Hamming panel ID differs from the sealed dataset".into());
        }
        let root = &record.state;
        let started = Instant::now();
        let result = search_hamming(root, manifest.search)?;
        let wall_seconds = started.elapsed().as_secs_f64();
        let identity = crate::MatrixState::identity(root.dimension(), root.alphabet_id())
            .map_err(|error| error.to_string())?;
        let solved = result.word.is_some();
        let end_packed_rows = solved.then(|| identity.packed_rows().to_vec());
        let word = result.word.map(|gates| {
            json!({
                "control_target": gates.iter().map(|gate| [gate.control, gate.target]).collect::<Vec<_>>(),
                "physical_cost": gates.len(),
                "replayed": true,
            })
        });
        rows.push(json!({
            "panel_index": index,
            "case_id": record.lineage_id,
            "dimension": root.dimension(),
            "width": root.dimension(),
            "alphabet_id": root.alphabet_id(),
            "start_packed_rows": root.packed_rows(),
            "end_packed_rows": end_packed_rows,
            "status": result.status,
            "word": word,
            "depths": result.depths,
            "expanded": result.expanded,
            "generated": result.generated,
            "value_evaluations": result.value_evaluations,
            "action_evaluations": result.action_evaluations,
            "value_slots": result.value_slots,
            "action_slots": result.action_slots,
            "wall_seconds": wall_seconds,
            "declaration_position": position,
        }));
    }

    Ok(json!({
        "schema": "cnot103_hamming_search_result_v1",
        "source_revision": manifest.source_revision,
        "search_manifest_sha256": sha256(&raw),
        "dataset_sha256": dataset.sha256,
        "panel": manifest.panel,
        "panel_indices": manifest.panel_indices,
        "panel_ids": manifest.panel_ids,
        "panel_roots": manifest.panel_roots,
        "panel_digest": manifest.panel_digest,
        "runtime_contract": manifest.runtime_contract,
        "search": manifest.search,
        "head": "hamming",
        "execution_backend": "cpu_hamming",
        "rows": rows,
    }))
}

