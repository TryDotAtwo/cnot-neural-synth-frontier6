//! Strict serde model for the sealed frozen-data JSON wire format.

use serde::Deserialize;
use serde_json::{Number, Value};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::frozen_data) struct Body {
    pub(in crate::frozen_data) schema: String,
    pub(in crate::frozen_data) provenance: Value,
    pub(in crate::frozen_data) symmetries: String,
    pub(in crate::frozen_data) forbidden_keys: Vec<Key>,
    pub(in crate::frozen_data) historical_keys: Vec<Key>,
    pub(in crate::frozen_data) rejected_train_lineages: Vec<String>,
    pub(in crate::frozen_data) panels: Panels,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::frozen_data) struct Panels {
    pub(in crate::frozen_data) train: Vec<Record>,
    pub(in crate::frozen_data) dev: Vec<Record>,
    pub(in crate::frozen_data) test: Vec<Record>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::frozen_data) struct State {
    pub(in crate::frozen_data) n: u32,
    pub(in crate::frozen_data) packed_rows: Vec<Number>,
    pub(in crate::frozen_data) alphabet_id: String,
}
pub(in crate::frozen_data) type Key = (u32, String, Vec<Number>);
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::frozen_data) struct Gate {
    pub(in crate::frozen_data) control: u32,
    pub(in crate::frozen_data) target: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::frozen_data) struct Word {
    pub(in crate::frozen_data) start: State,
    pub(in crate::frozen_data) end: State,
    pub(in crate::frozen_data) gates: Vec<Gate>,
    pub(in crate::frozen_data) physical_cost: u64,
    pub(in crate::frozen_data) producer: String,
    pub(in crate::frozen_data) lineage_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::frozen_data) struct Record {
    pub(in crate::frozen_data) state: State,
    pub(in crate::frozen_data) verified_words: Vec<Word>,
    pub(in crate::frozen_data) lineage_id: String,
    pub(in crate::frozen_data) seed: u64,
    pub(in crate::frozen_data) source: String,
    pub(in crate::frozen_data) split: String,
    #[serde(default)]
    pub(in crate::frozen_data) exact_distance: Option<u64>,
    #[serde(default)]
    pub(in crate::frozen_data) exact_evidence: Option<String>,
}
