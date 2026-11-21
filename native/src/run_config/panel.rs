//! Shared sealed dataset and ordered panel identity validation.
use crate::frozen_data::{FrozenDataset, FrozenPanel};
use crate::{CnotAlphabet, TrainingSplit};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PanelRootPin {
    pub(super) panel_index: usize,
    pub(super) case_id: String,
    pub(super) dimension: u32,
    pub(super) alphabet_id: String,
    pub(super) start_packed_rows: Vec<u64>,
}

pub(super) fn frozen_panel_digest(panel: &str, indices: &[usize], ids: &[String]) -> String {
    let value = json!({"panel": panel, "indices": indices, "ids": ids});
    let bytes = serde_json::to_vec(&value).expect("panel identity JSON is serializable");
    Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn validate_frozen_panel<'a>(
    dataset: &'a FrozenDataset,
    panel_name: &str,
    indices: &[usize],
    ids: &[String],
    digest: &str,
) -> Result<&'a FrozenPanel, String> {
    let (panel, split) = match panel_name {
        "dev" => (&dataset.dev, TrainingSplit::Dev),
        "test" => (&dataset.test, TrainingSplit::Test),
        _ => return Err("sealed search panel must be dev or test".into()),
    };
    if indices.is_empty()
        || indices.len() != ids.len()
        || indices.iter().collect::<BTreeSet<_>>().len() != indices.len()
        || ids.iter().any(|id| id.trim().is_empty())
        || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
        || frozen_panel_digest(panel_name, indices, ids) != digest
    {
        return Err("sealed search panel declaration is invalid".into());
    }
    for (position, (&index, id)) in indices.iter().zip(ids).enumerate() {
        let record = panel.records().get(index).ok_or("sealed panel index is out of range")?;
        let alphabet = CnotAlphabet::all_to_all_unit(record.state.dimension())
            .map_err(|error| error.to_string())?;
        if record.split != split
            || record.lineage_id != *id
            || record.state.alphabet_id() != alphabet.id()
            || !dataset.forbidden_keys.contains(&record.state)
        {
            return Err(format!("sealed panel root identity failed at declared position {position}"));
        }
    }
    Ok(panel)
}

pub(super) fn validate_panel_root_pins(
    panel: &FrozenPanel,
    indices: &[usize],
    ids: &[String],
    roots: &[PanelRootPin],
) -> Result<(), String> {
    if roots.len() != indices.len() || roots.len() != ids.len() {
        return Err("sealed panel root pin count differs from its declaration".into());
    }
    for (position, root) in roots.iter().enumerate() {
        let index = indices[position];
        let id = &ids[position];
        let record = panel.records().get(index).ok_or("sealed panel index is out of range")?;
        if root.panel_index != index
            || root.case_id != *id
            || root.dimension != record.state.dimension()
            || root.alphabet_id != record.state.alphabet_id()
            || root.start_packed_rows != record.state.packed_rows()
        {
            return Err(format!("sealed panel root pin failed at declared position {position}"));
        }
    }
    Ok(())
}

pub(super) fn load_pinned_panel(
    path: &Path,
    dataset_sha256: &str,
    panel_name: &str,
    indices: &[usize],
    ids: &[String],
    digest: &str,
) -> Result<FrozenDataset, String> {
    let dataset = FrozenDataset::load_pinned(path, dataset_sha256).map_err(|e| e.to_string())?;
    validate_frozen_panel(&dataset, panel_name, indices, ids, digest)?;
    Ok(dataset)
}
