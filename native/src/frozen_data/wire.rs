//! Private serde schema and semantic decoding for sealed frozen-data artifacts.
use super::alphabet::alphabet;
use super::types::{FrozenDataError, FrozenPanel, FrozenRecordMetadata};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use crate::training_data::{ExactCatalog, TrainingRecord, TrainingSplit};
use num_bigint::BigUint;
use serde::Deserialize;
use serde_json::{Number, Value};
use std::collections::{HashMap, HashSet};

const MAX_N: u32 = 256;

fn split_name(s: TrainingSplit) -> &'static str {
    match s {
        TrainingSplit::Train => "train",
        TrainingSplit::Dev => "dev",
        TrainingSplit::Test => "test",
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Body {
    pub(super) schema: String,
    pub(super) provenance: Value,
    pub(super) symmetries: String,
    pub(super) forbidden_keys: Vec<Key>,
    pub(super) historical_keys: Vec<Key>,
    pub(super) rejected_train_lineages: Vec<String>,
    pub(super) panels: Panels,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Panels {
    pub(super) train: Vec<Record>,
    pub(super) dev: Vec<Record>,
    pub(super) test: Vec<Record>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct State {
    pub(super) n: u32,
    pub(super) packed_rows: Vec<Number>,
    pub(super) alphabet_id: String,
}
pub(super) type Key = (u32, String, Vec<Number>);
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Gate {
    control: u32,
    target: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Word {
    start: State,
    end: State,
    gates: Vec<Gate>,
    physical_cost: u64,
    producer: String,
    lineage_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Record {
    state: State,
    verified_words: Vec<Word>,
    lineage_id: String,
    seed: u64,
    source: String,
    split: String,
    #[serde(default)]
    exact_distance: Option<u64>,
    #[serde(default)]
    exact_evidence: Option<String>,
}

pub(super) fn panel(
    raw: Vec<Record>,
    split: TrainingSplit,
    name: &str,
    catalogs: &HashMap<String, ExactCatalog>,
    units: &mut HashMap<(String, u64), TrainingSplit>,
    rejected: &HashSet<String>,
) -> Result<FrozenPanel, FrozenDataError> {
    let mut records = Vec::with_capacity(raw.len());
    let mut metadata = Vec::with_capacity(raw.len());
    let mut roots = HashSet::with_capacity(raw.len());
    for (index, item) in raw.into_iter().enumerate() {
        if item.split != name {
            return Err(FrozenDataError(format!("{name}[{index}] split mismatch")));
        }
        if item.source.trim().is_empty() || item.lineage_id.trim().is_empty() {
            return Err(FrozenDataError(format!(
                "{name}[{index}] lacks source/lineage"
            )));
        }
        if let Some(old) = units.insert((item.source.clone(), item.seed), split) {
            if old != split {
                return Err(FrozenDataError(format!(
                    "source/seed family leaks from {} to {name}",
                    split_name(old)
                )));
            }
        }
        let state = decode_state(item.state)?;
        if !roots.insert(state.clone()) {
            return Err(FrozenDataError(format!("{name} repeats a root")));
        }
        let alpha = alphabet(&state)?;
        if item.verified_words.is_empty() {
            return Err(FrozenDataError(format!("{name}[{index}] has no word")));
        }
        let identity = MatrixState::identity(alpha.dimension(), alpha.id())
            .map_err(|e| FrozenDataError(e.to_string()))?;
        let mut words = Vec::with_capacity(item.verified_words.len());
        for raw_word in item.verified_words {
            let start = decode_state(raw_word.start)?;
            let end = decode_state(raw_word.end)?;
            if start != state || end != identity {
                return Err(FrozenDataError(format!(
                    "{name}[{index}] word endpoints mismatch"
                )));
            }
            let cert = WordCertificate {
                start,
                end,
                gates: raw_word
                    .gates
                    .into_iter()
                    .map(|g| CnotGate {
                        control: g.control,
                        target: g.target,
                    })
                    .collect(),
                physical_cost: raw_word.physical_cost,
                producer: raw_word.producer,
                lineage_id: raw_word.lineage_id,
            };
            cert.verify(&alpha)
                .map_err(|e| FrozenDataError(e.to_string()))?;
            words.push(cert);
        }
        let record = TrainingRecord {
            state,
            split,
            verified_words: words,
            exact_distance: item.exact_distance,
            lineage_id: item.lineage_id,
        };
        let mut verifier = record.clone();
        verifier.split = TrainingSplit::Train;
        verifier
            .validate_for_training(&alpha, catalogs.get(alpha.id()))
            .map_err(|e| FrozenDataError(format!("{name}[{index}]: {e}")))?;
        if split == TrainingSplit::Train && rejected.contains(&record.lineage_id) {
            return Err(FrozenDataError(
                "accepted train row has rejected lineage".into(),
            ));
        }
        metadata.push(FrozenRecordMetadata {
            source: item.source,
            seed: item.seed,
            exact_evidence: item.exact_evidence,
        });
        records.push(record);
    }
    Ok(FrozenPanel { records, metadata })
}

pub(super) fn decode_state(raw: State) -> Result<MatrixState, FrozenDataError> {
    if raw.n == 0 || raw.n > MAX_N {
        return Err(FrozenDataError(format!(
            "unsupported matrix dimension {}",
            raw.n
        )));
    }
    let alpha = CnotAlphabet::all_to_all_unit(raw.n).map_err(|e| FrozenDataError(e.to_string()))?;
    if raw.alphabet_id != alpha.id() || raw.packed_rows.len() != raw.n as usize {
        return Err(FrozenDataError(
            "matrix alphabet or row count mismatch".into(),
        ));
    }
    let width = (raw.n as usize + 63) / 64;
    let mut packed = Vec::with_capacity(raw.n as usize * width);
    for (i, value) in raw.packed_rows.iter().enumerate() {
        let text = value.to_string();
        let bits = BigUint::parse_bytes(text.as_bytes(), 10)
            .ok_or_else(|| FrozenDataError(format!("row {i} is not a nonnegative integer")))?;
        if bits.bits() > u64::from(raw.n) {
            return Err(FrozenDataError(format!(
                "row {i} has bits outside matrix width"
            )));
        }
        let words = bits.to_u64_digits();
        for word in 0..width {
            packed.push(words.get(word).copied().unwrap_or(0));
        }
    }
    MatrixState::from_packed_rows(raw.n, raw.alphabet_id, packed)
        .map_err(|e| FrozenDataError(e.to_string()))
}

pub(super) fn key_set(keys: Vec<Key>, name: &str) -> Result<HashSet<MatrixState>, FrozenDataError> {
    let mut out = HashSet::with_capacity(keys.len());
    for (n, alphabet_id, packed_rows) in keys {
        let state = decode_state(State {
            n,
            alphabet_id,
            packed_rows,
        })
        .map_err(|e| FrozenDataError(format!("{name}: {e}")))?;
        if !out.insert(state) {
            return Err(FrozenDataError(format!("{name} has duplicate keys")));
        }
    }
    Ok(out)
}
