use cnot103_native::{
    CnotAlphabet, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole, FeatureStatus,
    MatrixState,
};
use serde_json::Value;
use std::collections::HashSet;

const PYTHON_ARCHIVE_SHA256: &str =
    "9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866";
const PYTHON_SCHEMA_SOURCE_SHA256: &str =
    "ef92e7bce36182d5562943702ef8371302d78fb5216532c91579f918def21f26";
const PYTHON_SCHEMA_MANIFEST_SHA256: &str =
    "3b4063047ccc9d337e74d92de90108653ac5cf9da166733eba2f4c429c48c0b8";
const PYTHON_STRUCTURAL_SHA256: &str =
    "fe1c479ceef39c6b8a64f078a6166993d31ef833c43f7ad78b5ed481f4821049";

fn parse_id(value: &Value) -> FeatureId {
    let text = value.as_str().expect("feature ID must be text");
    let bytes = text.as_bytes();
    assert!(bytes.len() >= 2, "invalid feature ID: {text}");
    let ordinal = text[1..]
        .parse::<u8>()
        .expect("feature ordinal must be numeric");
    FeatureId::new(bytes[0], ordinal).expect("feature ID must be canonical")
}

fn numbers(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("feature values must be a flat array")
        .iter()
        .map(|number| number.as_f64().expect("feature value must be numeric") as f32)
        .collect()
}

fn role(value: &Value) -> FeatureRole {
    match value.as_str().expect("feature role must be text") {
        "node" => FeatureRole::Node,
        "pair" => FeatureRole::Pair,
        "global" => FeatureRole::Global,
        "group" => FeatureRole::Group,
        unknown => panic!("unknown feature role: {unknown}"),
    }
}

#[test]
fn all_python_v17_source_backed_structural_features_match_rust() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/python_structural_33.json"))
        .expect("pinned Python structural fixture must be valid JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("python_structural_reference_v2")
    );
    assert_eq!(
        fixture["source_archive_sha256"].as_str(),
        Some(PYTHON_ARCHIVE_SHA256)
    );
    assert_eq!(
        fixture["feature_schema_source_sha256"].as_str(),
        Some(PYTHON_SCHEMA_SOURCE_SHA256)
    );
    assert_eq!(
        fixture["feature_schema_manifest_sha256"].as_str(),
        Some(PYTHON_SCHEMA_MANIFEST_SHA256)
    );
    assert_eq!(
        fixture["python_structural_sha256"].as_str(),
        Some(PYTHON_STRUCTURAL_SHA256)
    );

    let mut names = (1..=20).map(|i| format!("A{i:02}")).collect::<Vec<_>>();
    names.extend((1..=12).map(|i| format!("B{i:02}")));
    names.push("F01".into());
    assert_eq!(fixture["feature_ids"], serde_json::json!(names));

    let ids = names
        .iter()
        .map(|name| parse_id(&Value::String(name.clone())))
        .collect::<Vec<_>>();
    let mut total_states = 0usize;

    for batch in fixture["batches"]
        .as_array()
        .expect("fixture must contain width-specific batches")
    {
        let n = batch["n"].as_u64().expect("batch width must be numeric") as usize;
        assert!((3..=4).contains(&n));
        let expected_count = batch["state_count"]
            .as_u64()
            .expect("state count must be numeric") as usize;
        let expected_exhaustive_count = if n == 3 {
            168 // |GL(3, 2)| = (2^3 - 1)(2^3 - 2)(2^3 - 2^2)
        } else {
            5
        };
        assert_eq!(expected_count, expected_exhaustive_count);

        let alphabet = CnotAlphabet::all_to_all_unit(n as u32).expect("CNOT alphabet");
        let mut states = Vec::with_capacity(expected_count);
        let mut identities = HashSet::new();
        let mut state_names = HashSet::new();

        for row in batch["states"]
            .as_array()
            .expect("batch states must be an array")
        {
            assert_eq!(row["n"].as_u64(), Some(n as u64));
            let name = row["name"].as_str().expect("state name must be text");
            assert!(state_names.insert(name), "duplicate state name: {name}");
            let path = row["cnot_path"]
                .as_array()
                .expect("fixture state must contain a CNOT construction path");
            let mut regenerated = (0..n).map(|wire| 1u64 << wire).collect::<Vec<_>>();
            for gate in path {
                let coordinates = gate.as_array().expect("CNOT path item must be a pair");
                assert_eq!(coordinates.len(), 2);
                let control = coordinates[0].as_u64().unwrap() as usize;
                let target = coordinates[1].as_u64().unwrap() as usize;
                assert!(control < n && target < n && control != target);
                regenerated[target] ^= regenerated[control];
            }
            let packed_rows = row["packed_rows"]
                .as_array()
                .expect("packed_rows must be an array")
                .iter()
                .map(|word| word.as_u64().expect("packed row must be unsigned"))
                .collect::<Vec<_>>();
            assert_eq!(regenerated, packed_rows, "CNOT path mismatch for {name}");
            let state = MatrixState::from_packed_rows(n as u32, alphabet.id(), packed_rows)
                .expect("each state must belong to GL(n, 2)");
            assert!(identities.insert(state.clone()), "duplicate fixture matrix");
            states.push(state);
        }
        assert_eq!(states.len(), expected_count);
        total_states += states.len();

        let policy = FeaturePolicy {
            enabled_ids: ids.clone(),
            seed: 0,
            schema_id: "python-v17-structural-parity-v2".into(),
            solver_assisted: false,
            symmetry_contract: "exact_relabel".into(),
            evaluations: None,
            max_bytes: None,
        };
        let registry = FeatureRegistry::new();
        registry
            .selected_specs(&policy)
            .expect("all 33 source-backed structural providers must be executable");
        let packet = registry
            .build(&states, &policy)
            .expect("Rust must build every source-backed structural feature");

        let expected_features = batch["features"]
            .as_array()
            .expect("batch features must be an array");
        assert_eq!(expected_features.len(), names.len());
        for expected in expected_features {
            let id = parse_id(&expected["id"]);
            let actual = packet.feature(id);
            assert_eq!(actual.spec.role, role(&expected["role"]), "{id} role");
            assert_eq!(
                actual.spec.channels,
                expected["channels"].as_u64().unwrap() as usize,
                "{id} channels"
            );
            let mut expected_shape = vec![states.len()];
            expected_shape.extend(
                expected["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|extent| extent.as_u64().unwrap() as usize),
            );
            assert_eq!(actual.shape, expected_shape, "{id} shape");
            assert_eq!(actual.available, vec![1; states.len()], "{id} availability");
            assert!(
                actual
                    .status
                    .iter()
                    .all(|status| *status == FeatureStatus::Computed),
                "{id} status"
            );

            let reference = numbers(&expected["values"]);
            assert_eq!(actual.values.len(), reference.len(), "{id} value count");
            for (index, (actual, expected)) in actual.values.iter().zip(reference).enumerate() {
                let tolerance = 2e-6 * (1.0 + expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{id}[{index}]: Rust={actual:.9}, Python={expected:.9}, tolerance={tolerance:.3e}"
                );
            }
        }
    }

    assert_eq!(total_states, 173);
}
