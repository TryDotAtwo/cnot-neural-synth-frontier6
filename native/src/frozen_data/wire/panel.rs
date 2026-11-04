//! Semantic validation and conversion of frozen training panels.

use super::super::alphabet::alphabet;
use super::super::types::{FrozenDataError, FrozenPanel, FrozenRecordMetadata};
use super::model::Record;
use super::state::decode_state;
use crate::alphabet::WordCertificate;
use crate::state::{CnotGate, MatrixState};
use crate::training_data::{ExactCatalog, TrainingRecord, TrainingSplit};
use std::collections::{HashMap, HashSet};

fn split_name(s: TrainingSplit) -> &'static str {
    match s {
        TrainingSplit::Train => "train",
        TrainingSplit::Dev => "dev",
        TrainingSplit::Test => "test",
    }
}

pub(in crate::frozen_data) fn panel(
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
