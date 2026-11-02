use cnot103_native::training_data::load_exact_catalogs_pinned;
use cnot103_native::{CnotAlphabet, MatrixState};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn catalog_fixture() -> Value {
    json!({
        "schema": "exact_distance_catalog_bundle_v1",
        "provenance": {"source": "complete GL(2,2) test fixture"},
        "catalogs": [{
            "dimension": 2,
            "entries": [
                {"packed_rows": [1, 2], "distance": 0, "gates": [], "producer": "test", "lineage_id": "identity"},
                {"packed_rows": [1, 3], "distance": 1, "gates": [{"control": 0, "target": 1}], "producer": "test", "lineage_id": "a"},
                {"packed_rows": [3, 2], "distance": 1, "gates": [{"control": 1, "target": 0}], "producer": "test", "lineage_id": "b"},
                {"packed_rows": [2, 3], "distance": 2, "gates": [{"control": 1, "target": 0}, {"control": 0, "target": 1}], "producer": "test", "lineage_id": "c"},
                {"packed_rows": [3, 1], "distance": 2, "gates": [{"control": 0, "target": 1}, {"control": 1, "target": 0}], "producer": "test", "lineage_id": "d"},
                {"packed_rows": [2, 1], "distance": 3, "gates": [{"control": 1, "target": 0}, {"control": 0, "target": 1}, {"control": 1, "target": 0}], "producer": "test", "lineage_id": "e"}
            ]
        }]
    })
}

fn write_fixture(body: &Value) -> (PathBuf, String, PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after UNIX epoch")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "cnot103-exact-catalog-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&directory).expect("create test directory");
    let path = directory.join("catalog.json");
    let bytes = serde_json::to_vec(body).expect("serialize catalog fixture");
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    fs::write(&path, bytes).expect("write catalog");
    fs::write(path.with_extension("sha256"), &digest).expect("write catalog sidecar");
    (path, digest, directory)
}

#[test]
fn pinned_catalog_bundle_verifies_all_exact_distances() {
    let body = catalog_fixture();
    let (path, digest, directory) = write_fixture(&body);
    let catalogs = load_exact_catalogs_pinned(&path, &digest).expect("verify pinned catalog");
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let catalog = catalogs.get(alphabet.id()).expect("dimension catalog");
    assert_eq!(catalog.state_count(), 6);
    for (rows, distance) in [
        (vec![1, 2], 0),
        (vec![1, 3], 1),
        (vec![3, 2], 1),
        (vec![2, 3], 2),
        (vec![3, 1], 2),
        (vec![2, 1], 3),
    ] {
        let state = MatrixState::from_packed_rows(2, alphabet.id(), rows).unwrap();
        assert_eq!(catalog.exact_distance(&state), Some(distance));
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn catalog_bundle_requires_the_trusted_digest_and_matching_sidecar() {
    let (path, digest, directory) = write_fixture(&catalog_fixture());
    let wrong_pin = "0".repeat(64);
    assert!(load_exact_catalogs_pinned(&path, &wrong_pin).is_err());
    fs::write(path.with_extension("sha256"), "f".repeat(64)).unwrap();
    assert!(load_exact_catalogs_pinned(&path, &digest).is_err());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn catalog_bundle_rejects_a_nonreplayable_exact_label_certificate() {
    let mut body = catalog_fixture();
    body["catalogs"][0]["entries"][3]["distance"] = json!(3);
    let (path, digest, directory) = write_fixture(&body);
    assert!(load_exact_catalogs_pinned(&path, &digest).is_err());
    fs::remove_dir_all(directory).unwrap();
}
