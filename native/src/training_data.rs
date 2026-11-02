//! Verified training records and exact-distance catalogs.
//!
//! Exact labels are admitted only from a finite, alphabet-closed catalog with
//! replayable reductions and a globally 1-Lipschitz distance potential. This
//! is stronger than treating one short word as a shortest-path proof.

use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::CnotGate;
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingDataError(pub String);

impl Display for TrainingDataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TrainingDataError {}

/// Port of the source `cheap_bounds` function: nonzero rows/columns and GF(2)
/// rank of `M xor I`, combined by their maximum.
pub fn elementary_lower_bound(state: &MatrixState) -> Result<u64, TrainingDataError> {
    let n = usize::try_from(state.dimension())
        .map_err(|_| TrainingDataError("matrix dimension exceeds host index range".into()))?;
    let words_per_row = state.words_per_row();
    let mut residual = Vec::new();
    residual
        .try_reserve_exact(n)
        .map_err(|_| TrainingDataError("could not allocate lower-bound rows".into()))?;

    let mut changed_rows = 0usize;
    let mut changed_columns = vec![false; n];
    for row_index in 0..n {
        let start = row_index
            .checked_mul(words_per_row)
            .ok_or_else(|| TrainingDataError("matrix row offset overflow".into()))?;
        let mut row = state.packed_rows()[start..start + words_per_row].to_vec();
        row[row_index / 64] ^= 1u64 << (row_index % 64);
        changed_rows += usize::from(row.iter().any(|word| *word != 0));
        for (column, changed) in changed_columns.iter_mut().enumerate() {
            if ((row[column / 64] >> (column % 64)) & 1) != 0 {
                *changed = true;
            }
        }
        residual.push(row);
    }
    let rank = gf2_rank_packed(&mut residual, n);
    let changed_columns = changed_columns
        .into_iter()
        .filter(|changed| *changed)
        .count();
    Ok(changed_rows.max(changed_columns).max(rank) as u64)
}

fn gf2_rank_packed(rows: &mut [Vec<u64>], n: usize) -> usize {
    let mut pivots: Vec<Option<Vec<u64>>> = vec![None; n];
    let mut rank = 0usize;
    for row in rows {
        for column in (0..n).rev() {
            if ((row[column / 64] >> (column % 64)) & 1) == 0 {
                continue;
            }
            if let Some(pivot) = &pivots[column] {
                for (word, pivot_word) in row.iter_mut().zip(pivot) {
                    *word ^= pivot_word;
                }
            } else {
                pivots[column] = Some(row.clone());
                rank += 1;
                break;
            }
        }
    }
    rank
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingSplit {
    Train,
    Dev,
    Test,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactCatalogEntry {
    pub distance: u64,
    pub certificate: WordCertificate,
}

/// An exact-distance table whose proof obligations were checked at creation.
#[derive(Clone, Debug)]
pub struct ExactCatalog {
    alphabet_id: String,
    distances: HashMap<MatrixState, u64>,
    entries: HashMap<MatrixState, ExactCatalogEntry>,
    sha256: String,
}

impl ExactCatalog {
    /// Verify every state, reduction word, outgoing edge, and Lipschitz step
    /// before retaining the catalog as a source of exact labels.
    pub fn new(
        alphabet: &CnotAlphabet,
        entries: HashMap<MatrixState, ExactCatalogEntry>,
    ) -> Result<Self, TrainingDataError> {
        if entries.is_empty() {
            return Err(TrainingDataError("exact catalog is empty".into()));
        }
        let identity = MatrixState::identity(alphabet.dimension(), alphabet.id())
            .map_err(|error| TrainingDataError(error.to_string()))?;
        if entries.get(&identity).map(|entry| entry.distance) != Some(0) {
            return Err(TrainingDataError(
                "exact catalog must assign distance zero to identity".into(),
            ));
        }

        for (state, entry) in &entries {
            if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
                return Err(TrainingDataError(
                    "exact catalog contains a state from another alphabet".into(),
                ));
            }
            if entry.certificate.start != *state
                || entry.certificate.end != identity
                || entry.certificate.physical_cost != entry.distance
            {
                return Err(TrainingDataError(
                    "exact catalog reduction certificate does not match its state and distance"
                        .into(),
                ));
            }
            entry
                .certificate
                .verify(alphabet)
                .map_err(|error| TrainingDataError(error.to_string()))?;

            for (gate_index, gate) in alphabet.gates().iter().copied().enumerate() {
                let cost =
                    u64::from(alphabet.cost_at(gate_index).ok_or_else(|| {
                        TrainingDataError("alphabet gate cost is missing".into())
                    })?);
                let mut neighbor = state.clone();
                neighbor
                    .apply_cnot(gate)
                    .map_err(|error| TrainingDataError(error.to_string()))?;
                let neighbor_distance = entries
                    .get(&neighbor)
                    .ok_or_else(|| {
                        TrainingDataError("exact catalog is not closed under its alphabet".into())
                    })?
                    .distance;
                if entry.distance.abs_diff(neighbor_distance) > cost {
                    return Err(TrainingDataError(
                        "exact catalog distances violate the alphabet cost Lipschitz bound".into(),
                    ));
                }
            }
        }

        // A replayable path gives d(state) <= path_cost; the closed
        // 1-Lipschitz potential gives d(state) >= every path_cost to identity.
        // Equality of both values proves every stored distance exact.
        let distances: HashMap<_, _> = entries
            .iter()
            .map(|(state, entry)| (state.clone(), entry.distance))
            .collect();
        let sha256 = catalog_digest(alphabet.id(), &entries);
        Ok(Self {
            alphabet_id: alphabet.id().to_owned(),
            distances,
            entries,
            sha256,
        })
    }

    pub fn alphabet_id(&self) -> &str {
        &self.alphabet_id
    }

    pub fn exact_distance(&self, state: &MatrixState) -> Option<u64> {
        self.distances.get(state).copied()
    }

    pub fn state_count(&self) -> usize {
        self.distances.len()
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn entries(&self) -> &HashMap<MatrixState, ExactCatalogEntry> {
        &self.entries
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingRecord {
    pub state: MatrixState,
    pub split: TrainingSplit,
    pub verified_words: Vec<WordCertificate>,
    pub exact_distance: Option<u64>,
    pub lineage_id: String,
}

impl TrainingRecord {
    /// Validate training-only provenance and return its best verified upper bound.
    pub fn validate_for_training(
        &self,
        alphabet: &CnotAlphabet,
        exact_catalog: Option<&ExactCatalog>,
    ) -> Result<u64, TrainingDataError> {
        if self.split != TrainingSplit::Train {
            return Err(TrainingDataError(
                "training record must belong to the train split".into(),
            ));
        }
        if self.lineage_id.trim().is_empty() {
            return Err(TrainingDataError(
                "training record lineage identity is required".into(),
            ));
        }
        if self.state.dimension() != alphabet.dimension()
            || self.state.alphabet_id() != alphabet.id()
        {
            return Err(TrainingDataError(
                "training record state does not match its alphabet".into(),
            ));
        }
        if self.verified_words.is_empty() {
            return Err(TrainingDataError(
                "training record has no verified upper-bound word".into(),
            ));
        }

        let identity = MatrixState::identity(alphabet.dimension(), alphabet.id())
            .map_err(|error| TrainingDataError(error.to_string()))?;
        let mut upper_bound = u64::MAX;
        for certificate in &self.verified_words {
            if certificate.start != self.state || certificate.end != identity {
                return Err(TrainingDataError(
                    "upper-bound word is not a reduction of the training state".into(),
                ));
            }
            certificate
                .verify(alphabet)
                .map_err(|error| TrainingDataError(error.to_string()))?;
            upper_bound = upper_bound.min(certificate.physical_cost);
        }
        let lower_bound = elementary_lower_bound(&self.state)?;
        if lower_bound > upper_bound {
            return Err(TrainingDataError(
                "verified word upper bound is below the elementary lower bound".into(),
            ));
        }

        if let Some(exact_distance) = self.exact_distance {
            if exact_distance < lower_bound || exact_distance > upper_bound {
                return Err(TrainingDataError(
                    "exact label lies outside its verified lower/upper bounds".into(),
                ));
            }
            let proved_by_bounds = lower_bound == exact_distance && upper_bound == exact_distance;
            if !proved_by_bounds {
                let catalog = exact_catalog.ok_or_else(|| {
                    TrainingDataError(
                        "exact label needs a bound-equality proof or verified exact catalog".into(),
                    )
                })?;
                if catalog.alphabet_id() != alphabet.id()
                    || catalog.exact_distance(&self.state) != Some(exact_distance)
                {
                    return Err(TrainingDataError(
                        "exact label differs from the verified exact catalog".into(),
                    ));
                }
            } else if exact_catalog.is_some_and(|catalog| {
                catalog.alphabet_id() != alphabet.id()
                    || catalog.exact_distance(&self.state) != Some(exact_distance)
            }) {
                return Err(TrainingDataError(
                    "exact label differs from the verified exact catalog".into(),
                ));
            }
        }
        Ok(upper_bound)
    }
}

fn catalog_digest(alphabet_id: &str, entries: &HashMap<MatrixState, ExactCatalogEntry>) -> String {
    let mut rows: Vec<Vec<u8>> = entries
        .iter()
        .map(|(state, entry)| {
            let mut row = Vec::new();
            row.extend_from_slice(&state.dimension().to_le_bytes());
            append_bytes(&mut row, state.alphabet_id().as_bytes());
            row.extend_from_slice(&(state.packed_rows().len() as u64).to_le_bytes());
            for word in state.packed_rows() {
                row.extend_from_slice(&word.to_le_bytes());
            }
            row.extend_from_slice(&entry.distance.to_le_bytes());
            row
        })
        .collect();
    rows.sort();

    let mut digest = Sha256::new();
    append_digest_bytes(&mut digest, alphabet_id.as_bytes());
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        append_digest_bytes(&mut digest, &row);
    }
    let bytes: [u8; 32] = digest.finalize().into();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn append_bytes(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

fn append_digest_bytes(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

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
