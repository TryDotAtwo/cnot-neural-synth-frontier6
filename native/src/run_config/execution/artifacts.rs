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
    pub(super) exact_catalogs: PathBuf,
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
    let checkpoint_dir = resolve_path(manifest_path, &manifest.checkpoint_dir);
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

    Ok(RunPaths {
        run_sha256: sha256(raw_manifest),
        dataset: resolve_path(manifest_path, &manifest.dataset.path),
        exact_catalogs: resolve_path(manifest_path, &manifest.exact_catalogs.path),
        checkpoint_dir,
    })
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
            return Err(format!("pinned exact-catalog bundle supports only all-to-all unit alphabet {}; supplied {}", canonical.id(), alphabet.id()));
        }
    }

    let catalogs =
        load_exact_catalogs_pinned(&paths.exact_catalogs, &manifest.exact_catalogs.sha256)
            .map_err(|e| e.to_string())?;
    if catalogs.len() != alphabets.len() || alphabets.iter().any(|a| !catalogs.contains_key(a.id()))
    {
        return Err("exact catalogs must cover exactly the configured alphabets".into());
    }
    let dataset = FrozenDataset::load_pinned_with_catalogs(
        &paths.dataset,
        &manifest.dataset.sha256,
        &catalogs,
    )
    .map_err(|e| e.to_string())?;

    Ok(PinnedRunData {
        alphabets,
        catalogs,
        dataset,
    })
}

fn resolve_path(manifest: &Path, configured: &Path) -> PathBuf {
    if configured.is_absolute() {
        configured.to_owned()
    } else {
        manifest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(configured)
    }
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
