//! Public facade for sealed frozen-data loading and validation.
mod alphabet;
mod loader;
mod reader;
mod symmetry;
mod types;
mod wire;

pub use types::{FrozenDataError, FrozenDataset, FrozenPanel, FrozenRecordMetadata};

#[cfg(feature = "torch")]
pub fn write_cpu_hamming_fixture(path: impl AsRef<std::path::Path>) -> Result<String, String> {
    use crate::alphabet::CnotAlphabet;
    use crate::state::MatrixState;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;
    use std::path::Path;

    fn wire_state(state: &MatrixState) -> Value {
        let rows = state
            .packed_rows()
            .chunks_exact(state.words_per_row())
            .map(|row| row[0])
            .collect::<Vec<_>>();
        json!({
            "n": state.dimension(),
            "packed_rows": rows,
            "alphabet_id": state.alphabet_id(),
        })
    }

    fn fixture_record(n: u32, split: &str, seed: u64) -> Result<(Value, MatrixState), String> {
        let alphabet = CnotAlphabet::all_to_all_unit(n).map_err(|e| e.to_string())?;
        let identity = MatrixState::identity(n, alphabet.id()).map_err(|e| e.to_string())?;
        let mut rows = identity.packed_rows().to_vec();
        let words = identity.words_per_row();
        for word in 0..words {
            rows[words + word] ^= rows[word];
        }
        let root = MatrixState::from_packed_rows(n, alphabet.id(), rows)
            .map_err(|e| e.to_string())?;
        let lineage_id = format!("ci-{split}-n{n}");
        let record = json!({
            "state": wire_state(&root),
            "verified_words": [{
                "start": wire_state(&root),
                "end": wire_state(&identity),
                "gates": [{"control": 0, "target": 1}],
                "physical_cost": 1,
                "producer": "cpu_hamming_fixture_v1",
                "lineage_id": lineage_id,
            }],
            "lineage_id": lineage_id,
            "seed": seed,
            "source": format!("cpu-hamming-{split}"),
            "split": split,
            "exact_distance": null,
            "exact_evidence": null,
        });
        Ok((record, root))
    }

    fn key_json(state: &MatrixState) -> Value {
        let rows = state
            .packed_rows()
            .chunks_exact(state.words_per_row())
            .map(|row| row[0])
            .collect::<Vec<_>>();
        json!([state.dimension(), state.alphabet_id(), rows])
    }

    let path = path.as_ref();
    let (train, _) = fixture_record(4, "train", 401)?;
    let (dev, dev_state) = fixture_record(3, "dev", 301)?;
    let (test, test_state) = fixture_record(2, "test", 201)?;
    let mut forbidden = HashSet::new();
    forbidden.extend(symmetry::views(&dev_state).map_err(|e| e.to_string())?);
    forbidden.extend(symmetry::views(&test_state).map_err(|e| e.to_string())?);
    let body = json!({
        "schema": "own_frozen_panels_v1",
        "provenance": {"fixture": "cpu_hamming_fixture_v1"},
        "symmetries": "identity,inverse,transpose,inverse_transpose and wire reversal; not full orbit",
        "forbidden_keys": forbidden.iter().map(key_json).collect::<Vec<_>>(),
        "historical_keys": [],
        "rejected_train_lineages": [],
        "panels": {"train": [train], "dev": [dev], "test": [test]},
    });
    let bytes = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
    std::fs::write(path, &bytes).map_err(|e| e.to_string())?;
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    std::fs::write(path.with_extension("sha256"), format!("{digest}\n"))
        .map_err(|e| e.to_string())?;
    FrozenDataset::load_pinned(path, &digest).map_err(|e| e.to_string())?;
    Ok(digest)
}

#[cfg(test)]
#[path = "frozen_data/tests.rs"]
mod tests;
