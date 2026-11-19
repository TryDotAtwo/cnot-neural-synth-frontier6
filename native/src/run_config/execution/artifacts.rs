//! Run-file resolution and loading of checksum-pinned training inputs.

use super::super::manifest::RunManifest;
use crate::alphabet::CnotAlphabet;
use crate::frozen_data::FrozenDataset;
use crate::{load_exact_catalogs_pinned, ExactCatalog};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) struct RunPaths {
    pub(super) run_sha256: String,
    pub(super) dataset: PathBuf,
    manifest_path: PathBuf,
    pub(super) checkpoint_dir: PathBuf,
}

pub(super) struct PinnedRunData {
    pub(super) alphabets: Vec<CnotAlphabet>,
    pub(super) catalogs: std::collections::HashMap<String, ExactCatalog>,
    pub(super) dataset: FrozenDataset,
}

pub(super) fn resolve(
    manifest_path: &Path,
    raw_manifest: &[u8],
    manifest: &RunManifest,
) -> Result<RunPaths, String> {
    let paths = input_paths(manifest_path, raw_manifest, manifest);
    let checkpoint_dir = &paths.checkpoint_dir;
    if manifest.resume {
        if !checkpoint_dir.join("last.json").is_file() {
            return Err(format!(
                "resume requested but {} has no last.json",
                checkpoint_dir.display()
            ));
        }
    } else if checkpoint_dir.exists()
        && fs::read_dir(&checkpoint_dir)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "fresh run refuses nonempty checkpoint directory {}",
            checkpoint_dir.display()
        ));
    }

    Ok(paths)
}

pub(super) fn input_paths(
    manifest_path: &Path,
    raw_manifest: &[u8],
    manifest: &RunManifest,
) -> RunPaths {
    RunPaths {
        run_sha256: sha256(raw_manifest),
        dataset: resolve_path(manifest_path, &manifest.dataset.path),
        manifest_path: manifest_path.to_owned(),
        checkpoint_dir: resolve_path(manifest_path, &manifest.checkpoint_dir),
    }
}

pub(super) fn load(manifest: &RunManifest, paths: &RunPaths) -> Result<PinnedRunData, String> {
    let alphabets = manifest
        .alphabets
        .iter()
        .map(|spec| spec.build())
        .collect::<Result<Vec<_>, _>>()?;
    let mut ids = HashSet::with_capacity(alphabets.len());
    for alphabet in &alphabets {
        if !ids.insert(alphabet.id().to_owned()) {
            return Err(format!("duplicate alphabet identity {}", alphabet.id()));
        }
        let canonical =
            CnotAlphabet::all_to_all_unit(alphabet.dimension()).map_err(|e| e.to_string())?;
        if alphabet.id() != canonical.id()
            || alphabet.gates() != canonical.gates()
            || (0..alphabet.gates().len()).any(|i| alphabet.cost_at(i) != canonical.cost_at(i))
        {
            return Err(format!(
                "training requires the canonical all-to-all unit alphabet {}; supplied {}",
                canonical.id(),
                alphabet.id()
            ));
        }
    }

    let catalogs = match &manifest.exact_catalogs {
        Some(pin) => crate::stage_profile::measure("exact_catalog_load", || {
            load_exact_catalogs_pinned(resolve_path(&paths.manifest_path, &pin.path), &pin.sha256)
        })
        .map_err(|e| e.to_string())?,
        None => Default::default(),
    };
    if catalogs.keys().any(|id| !ids.contains(id)) {
        return Err("exact catalog uses an unconfigured alphabet".into());
    }
    let dataset = crate::stage_profile::measure("dataset_load", || {
        FrozenDataset::load_pinned_with_catalogs(
            &paths.dataset,
            &manifest.dataset.sha256,
            &catalogs,
        )
    })
    .map_err(|e| e.to_string())?;

    for panel in [&dataset.train, &dataset.dev, &dataset.test] {
        if panel
            .records()
            .iter()
            .any(|r| !ids.contains(r.state.alphabet_id()))
        {
            return Err("dataset record uses an unconfigured alphabet".into());
        }
    }
    Ok(PinnedRunData {
        alphabets,
        catalogs,
        dataset,
    })
}

pub(super) fn resolve_path(manifest: &Path, configured: &Path) -> PathBuf {
    if configured.is_absolute() {
        configured.to_owned()
    } else {
        manifest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(configured)
    }
}

pub(super) fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
