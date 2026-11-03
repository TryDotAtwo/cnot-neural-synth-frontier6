use super::loader::{SCHEMA, SYMMETRIES};
use super::symmetry::views;
use super::wire::{decode_state, State};
use super::FrozenDataset;
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use num_bigint::BigUint;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::str::FromStr;

fn one_gate_record(n: u32, split: &str) -> (Value, MatrixState) {
    let alpha = CnotAlphabet::all_to_all_unit(n).unwrap();
    let identity = MatrixState::identity(n, alpha.id()).unwrap();
    let gate = alpha.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let state_rows: Vec<_> = state
        .packed_rows()
        .chunks(state.words_per_row())
        .map(|row| {
            let bits = row
                .iter()
                .enumerate()
                .fold(BigUint::from(0u8), |acc, (word, bits)| {
                    acc | (BigUint::from(*bits) << (word * 64))
                });
            Number::from_str(&bits.to_str_radix(10)).unwrap()
        })
        .collect();
    let identity_rows: Vec<_> = identity
        .packed_rows()
        .chunks(identity.words_per_row())
        .map(|row| {
            let bits = row
                .iter()
                .enumerate()
                .fold(BigUint::from(0u8), |acc, (word, bits)| {
                    acc | (BigUint::from(*bits) << (word * 64))
                });
            Number::from_str(&bits.to_str_radix(10)).unwrap()
        })
        .collect();
    let lineage = format!("{split}-lineage");
    let record = json!({
        "state": {"n": n, "packed_rows": state_rows, "alphabet_id": alpha.id()},
        "verified_words": [{
            "start": {"n": n, "packed_rows": state.packed_rows().chunks(state.words_per_row()).map(|row| {
                let bits = row.iter().enumerate().fold(BigUint::from(0u8), |acc, (word, bits)| acc | (BigUint::from(*bits) << (word * 64)));
                Number::from_str(&bits.to_str_radix(10)).unwrap()
            }).collect::<Vec<_>>(), "alphabet_id": alpha.id()},
            "end": {"n": n, "packed_rows": identity_rows, "alphabet_id": alpha.id()},
            "gates": [{"control": gate.control, "target": gate.target}],
            "physical_cost": 1, "producer": "frozen-data-fixture", "lineage_id": lineage
        }],
        "lineage_id": lineage, "seed": n, "source": format!("source-{split}"),
        "split": split, "exact_distance": null, "exact_evidence": null
    });
    (record, state)
}

fn key_json(state: &MatrixState) -> Value {
    let rows: Vec<_> = state
        .packed_rows()
        .chunks(state.words_per_row())
        .map(|row| {
            let bits = row
                .iter()
                .enumerate()
                .fold(BigUint::from(0u8), |acc, (word, bits)| {
                    acc | (BigUint::from(*bits) << (word * 64))
                });
            Number::from_str(&bits.to_str_radix(10)).unwrap()
        })
        .collect();
    json!([state.dimension(), state.alphabet_id(), rows])
}

fn sealed_fixture(path: &Path) -> String {
    let (train, _) = one_gate_record(4, "train");
    let (dev, dev_state) = one_gate_record(3, "dev");
    let (test, test_state) = one_gate_record(2, "test");
    let mut forbidden = views(&dev_state).unwrap();
    forbidden.extend(views(&test_state).unwrap());
    let body = json!({
        "schema": SCHEMA,
        "provenance": {"fixture": "frozen_data_reader_test"},
        "symmetries": SYMMETRIES,
        "forbidden_keys": forbidden.iter().map(key_json).collect::<Vec<_>>(),
        "historical_keys": [],
        "rejected_train_lineages": [],
        "panels": {"train": [train], "dev": [dev], "test": [test]}
    });
    let bytes = serde_json::to_vec(&body).unwrap();
    std::fs::write(path, &bytes).unwrap();
    let digest = bytes
        .iter()
        .fold(Sha256::new(), |mut h, byte| {
            h.update([*byte]);
            h
        })
        .finalize()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect::<String>();
    std::fs::write(path.with_extension("sha256"), format!("{digest}\n")).unwrap();
    digest
}

#[test]
fn pinned_load_verifies_split_artifact_and_rejects_wrong_pin() {
    let unique = format!(
        "tqp-frozen-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let path = std::env::temp_dir().join(unique);
    let digest = sealed_fixture(&path);
    let data = FrozenDataset::load_pinned(&path, &digest).unwrap();
    assert_eq!(data.sha256, digest);
    assert_eq!(data.train.len(), 1);
    assert_eq!(data.dev.len(), 1);
    assert_eq!(data.test.len(), 1);
    assert!(FrozenDataset::load_pinned(&path, &"0".repeat(64)).is_err());
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(path.with_extension("sha256")).unwrap();
}

#[cfg(feature = "torch")]
#[test]
fn objective_binding_uses_the_loaded_artifact_and_its_train_panel() {
    use crate::features::{FeaturePolicy, FeatureRegistry};
    use crate::objective::{FullGraphObjective, FullGraphObjectiveConfig};

    let unique = format!(
        "tqp-frozen-bind-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let path = std::env::temp_dir().join(unique);
    let pinned = sealed_fixture(&path);
    let dataset = FrozenDataset::load_pinned(&path, &pinned).unwrap();
    let registry = FeatureRegistry::new();
    let policy = FeaturePolicy::all_model_inputs("frozen-data-binding-fixture");
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut objective = FullGraphObjective::new(
        vec![alphabet],
        &registry,
        policy,
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();

    objective.bind_frozen_dataset(&dataset).unwrap();
    let contract = objective.checkpoint_contract();
    assert_eq!(
        contract["frozen_dataset_sha256"].as_str(),
        Some(pinned.as_str())
    );
    assert!(contract["training_records_sha256"].as_str().is_some());
    assert!(contract["objective_sha256"].as_str().is_some());
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(path.with_extension("sha256")).unwrap();
}

#[test]
fn packed_rows_above_u64_keep_all_bits() {
    let n = 130u32;
    let a = CnotAlphabet::all_to_all_unit(n).unwrap();
    let rows: Vec<String> = (0..n)
        .map(|i| (BigUint::from(1u8) << i as usize).to_str_radix(10))
        .collect();
    let json = format!(
        "{{\"n\":{n},\"packed_rows\":[{}],\"alphabet_id\":\"{}\"}}",
        rows.join(","),
        a.id()
    );
    let raw: State = serde_json::from_str(&json).unwrap();
    let state = decode_state(raw).unwrap();
    assert_eq!(state.words_per_row(), 3);
    assert!(state.is_identity());
}
#[test]
fn rejects_nonzero_padding_bits() {
    let raw: State = serde_json::from_str(
        r#"{"n":3,"packed_rows":[1,2,8],"alphabet_id":"all_to_all_unit_cnot:3"}"#,
    )
    .unwrap();
    assert!(decode_state(raw)
        .unwrap_err()
        .0
        .contains("outside matrix width"));
}
