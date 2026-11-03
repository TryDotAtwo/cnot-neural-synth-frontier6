use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use super::{ExactCatalog, ExactCatalogEntry, TrainingDataError};

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

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactCatalogBundleBody {
    schema: String,
    provenance: serde_json::Value,
    catalogs: Vec<ExactCatalogWidth>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactCatalogWidth {
    dimension: u32,
    entries: Vec<ExactCatalogRawEntry>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactCatalogRawEntry {
    packed_rows: Vec<serde_json::Number>,
    distance: u64,
    gates: Vec<ExactCatalogRawGate>,
    producer: String,
    lineage_id: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactCatalogRawGate {
    control: u32,
    target: u32,
}

fn decode_catalog_state(
    dimension: u32,
    alphabet_id: &str,
    packed_rows: Vec<serde_json::Number>,
) -> Result<MatrixState, String> {
    if packed_rows.len() != dimension as usize {
        return Err("packed row count differs from matrix dimension".into());
    }
    let words_per_row = (dimension as usize)
        .checked_add(63)
        .ok_or_else(|| "packed row width overflow".to_owned())?
        / 64;
    let mut words = Vec::with_capacity(
        (dimension as usize)
            .checked_mul(words_per_row)
            .ok_or_else(|| "packed matrix size overflow".to_owned())?,
    );
    for (row_index, value) in packed_rows.iter().enumerate() {
        let text = value.to_string();
        let bits = num_bigint::BigUint::parse_bytes(text.as_bytes(), 10)
            .ok_or_else(|| format!("row {row_index} is not a nonnegative integer"))?;
        if bits.bits() > u64::from(dimension) {
            return Err(format!("row {row_index} has bits outside matrix width"));
        }
        let row_words = bits.to_u64_digits();
        for word_index in 0..words_per_row {
            words.push(row_words.get(word_index).copied().unwrap_or(0));
        }
    }
    MatrixState::from_packed_rows(dimension, alphabet_id, words).map_err(|error| error.to_string())
}

fn catalog_provenance_is_present(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        serde_json::Value::String(value) => !value.trim().is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
    }
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

struct CatalogHashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R> CatalogHashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finish(mut self) -> Result<String, TrainingDataError>
    where
        R: std::io::Read,
    {
        let mut buffer = [0u8; 1 << 16];
        loop {
            if std::io::Read::read(&mut self, &mut buffer).map_err(|error| {
                TrainingDataError(format!("finish hashing exact catalog: {error}"))
            })? == 0
            {
                break;
            }
        }
        Ok(self
            .hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

impl<R: std::io::Read> std::io::Read for CatalogHashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..count]);
        Ok(count)
    }
}
