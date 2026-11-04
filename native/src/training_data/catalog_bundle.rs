use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use std::collections::HashMap;

use super::{ExactCatalog, ExactCatalogEntry, TrainingDataError};
mod decode;
mod hashing;
mod raw;

use decode::{catalog_provenance_is_present, decode_catalog_state, is_sha256_digest};
use hashing::CatalogHashingReader;
use raw::ExactCatalogBundleBody;

const EXACT_CATALOG_BUNDLE_SCHEMA: &str = "exact_distance_catalog_bundle_v1";
const MAX_EXACT_CATALOG_DIMENSION: u32 = 256;

/// Load proof-checked exact-distance catalogs from a pinned bundle.
///
/// The bundle supplies one catalog per matrix width. Each row is encoded as a
/// nonnegative JSON integer, each certificate lists directed CNOT gates in
/// replay order, and each group uses the unit-cost all-to-all CNOT alphabet.
/// Both the adjacent .sha256 sidecar and the caller's trusted digest must match
/// the bytes before any catalog is admitted.
pub fn load_exact_catalogs_pinned(
    path: impl AsRef<std::path::Path>,
    trusted_sha256: &str,
) -> Result<HashMap<String, ExactCatalog>, TrainingDataError> {
    use std::io::BufReader;

    if !is_sha256_digest(trusted_sha256) {
        return Err(TrainingDataError(
            "trusted exact-catalog digest must be lowercase SHA-256 hex".into(),
        ));
    }
    let path = path.as_ref();
    let sidecar_path = path.with_extension("sha256");
    let sidecar = std::fs::read_to_string(&sidecar_path).map_err(|error| {
        TrainingDataError(format!(
            "could not read exact-catalog sidecar {}: {error}",
            sidecar_path.display()
        ))
    })?;
    let sealed_sha256 = sidecar.trim();
    if !is_sha256_digest(sealed_sha256) {
        return Err(TrainingDataError(
            "exact-catalog sidecar must contain lowercase SHA-256 hex".into(),
        ));
    }

    let file = std::fs::File::open(path)
        .map_err(|error| TrainingDataError(format!("open exact-catalog bundle: {error}")))?;
    let mut reader = CatalogHashingReader::new(BufReader::new(file));
    let body: ExactCatalogBundleBody = serde_json::from_reader(&mut reader)
        .map_err(|error| TrainingDataError(format!("invalid exact-catalog JSON: {error}")))?;
    let actual_sha256 = reader.finish()?;
    if sealed_sha256 != actual_sha256 {
        return Err(TrainingDataError(format!(
            "exact-catalog SHA-256 mismatch: sidecar says {sealed_sha256}, got {actual_sha256}"
        )));
    }
    if trusted_sha256 != actual_sha256 {
        return Err(TrainingDataError(format!(
            "exact-catalog digest differs from trusted pin: expected {trusted_sha256}, got {actual_sha256}"
        )));
    }
    if body.schema != EXACT_CATALOG_BUNDLE_SCHEMA {
        return Err(TrainingDataError(
            "exact-catalog bundle schema mismatch".into(),
        ));
    }
    if !catalog_provenance_is_present(&body.provenance) {
        return Err(TrainingDataError(
            "exact-catalog bundle provenance must be nonempty".into(),
        ));
    }
    if body.catalogs.is_empty() {
        return Err(TrainingDataError(
            "exact-catalog bundle must contain at least one width".into(),
        ));
    }

    let mut catalogs = HashMap::with_capacity(body.catalogs.len());
    for raw_catalog in body.catalogs {
        if raw_catalog.dimension == 0 || raw_catalog.dimension > MAX_EXACT_CATALOG_DIMENSION {
            return Err(TrainingDataError(format!(
                "unsupported exact-catalog matrix dimension {}",
                raw_catalog.dimension
            )));
        }
        let alphabet = CnotAlphabet::all_to_all_unit(raw_catalog.dimension)
            .map_err(|error| TrainingDataError(error.to_string()))?;
        if raw_catalog.entries.is_empty() {
            return Err(TrainingDataError(
                "exact catalog for a width must be nonempty".into(),
            ));
        }
        let identity = MatrixState::identity(raw_catalog.dimension, alphabet.id())
            .map_err(|error| TrainingDataError(error.to_string()))?;
        let mut entries = HashMap::with_capacity(raw_catalog.entries.len());
        for (index, raw_entry) in raw_catalog.entries.into_iter().enumerate() {
            let state =
                decode_catalog_state(raw_catalog.dimension, alphabet.id(), raw_entry.packed_rows)
                    .map_err(|error| TrainingDataError(format!("catalog entry {index}: {error}")))?;
            let certificate = WordCertificate {
                start: state.clone(),
                end: identity.clone(),
                gates: raw_entry
                    .gates
                    .into_iter()
                    .map(|gate| CnotGate {
                        control: gate.control,
                        target: gate.target,
                    })
                    .collect(),
                physical_cost: raw_entry.distance,
                producer: raw_entry.producer,
                lineage_id: raw_entry.lineage_id,
            };
            let entry = ExactCatalogEntry {
                distance: raw_entry.distance,
                certificate,
            };
            if entries.insert(state, entry).is_some() {
                return Err(TrainingDataError(
                    "exact catalog repeats a matrix state".into(),
                ));
            }
        }

        let catalog = ExactCatalog::new(&alphabet, entries)
            .map_err(|error| TrainingDataError(error.to_string()))?;
        let key = alphabet.id().to_owned();
        if catalogs.insert(key, catalog).is_some() {
            return Err(TrainingDataError(
                "exact-catalog bundle repeats a matrix width".into(),
            ));
        }
    }
    Ok(catalogs)
}
