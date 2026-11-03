use cnot103_native::diffusion::{register_diffusion, DiffusionConfig};
use cnot103_native::{
    register_basis_gradient, CnotAlphabet, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole,
    FeatureStatus, MatrixState,
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
const PYTHON_SUBSPACE_SHA256: &str =
    "78c06d9e542016f4cb0875f7a1a9807e8a638de6eebeddc7aadff57cd3084a99";
const PYTHON_GF2_SHA256: &str = "223d7bc72029457e8119946c6cd320c95063559ff63a61d1c67a9e7293bc21f2";
const PYTHON_PACKET_SHA256: &str =
    "b4e8b22065a7ec45a2de8c2b4d2a5e75f527abf4e131a7b9409951d1fa7341f1";
const PYTHON_BASIS_GRADIENT_SHA256: &str =
    "d450f15ed32454bff2dd069c66ccf3cb6395667f8828e1a563d522006bbc96e2";
const PYTHON_BASIS_MACROS_SHA256: &str =
    "ed34348949806453dd1c70daf957ee071c1cd445de787b70dc5346dd38840b53";

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

fn states_from_batch(batch: &Value, exhaustive: bool) -> Vec<MatrixState> {
    let n = batch["n"].as_u64().expect("batch width must be numeric") as usize;
    assert!((2..=6).contains(&n));
    let expected_count = batch["state_count"]
        .as_u64()
        .expect("state count must be numeric") as usize;
    let fixture_rows = batch["states"]
        .as_array()
        .expect("fixture states must be an array");
    assert_eq!(fixture_rows.len(), expected_count);
    if exhaustive {
        let expected_exhaustive_count = match n {
            3 => 168,
            4 => 5,
            _ => panic!("unexpected exhaustive width: {n}"),
        };
        assert_eq!(expected_count, expected_exhaustive_count);
    }

    let alphabet = CnotAlphabet::all_to_all_unit(n as u32).expect("CNOT alphabet");
    let mut states = Vec::with_capacity(expected_count);
    let mut identities = HashSet::new();
    let mut state_names = HashSet::new();

    for row in fixture_rows {
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
    states
}

fn assert_feature_rows(
    rows: &[Value],
    states: &[MatrixState],
    schema_id: &str,
    symmetry_contract: &str,
    register_basis: bool,
) {
    let ids = rows
        .iter()
        .map(|row| parse_id(&row["id"]))
        .collect::<Vec<_>>();
    let policy = FeaturePolicy {
        enabled_ids: ids,
        seed: 0,
        schema_id: schema_id.into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes: None,
    };
    let mut registry = FeatureRegistry::new();
    if register_basis {
        register_basis_gradient(&mut registry).expect("source basis-gradient providers");
    }
    registry
        .selected_specs(&policy)
        .expect("every source-backed provider must be executable");
    let packet = registry
        .build(states, &policy)
        .expect("Rust must build every source-backed feature");

    for expected in rows {
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

    for batch in fixture["batches"]
        .as_array()
        .expect("fixture must contain width-specific batches")
    {
        let states = states_from_batch(batch, true);
        assert_feature_rows(
            batch["features"].as_array().unwrap(),
            &states,
            "python-v17-structural-parity-v2",
            "exact_relabel",
            false,
        );
    }
}

#[test]
fn all_python_v17_opt_in_subspace_views_match_rust() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/python_subspace_views_6.json"))
            .expect("pinned Python subspace fixture must be valid JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("python_subspace_reference_v1")
    );
    assert_eq!(
        fixture["source_archive_sha256"].as_str(),
        Some(PYTHON_ARCHIVE_SHA256)
    );
    assert_eq!(
        fixture["subspace_views_sha256"].as_str(),
        Some(PYTHON_SUBSPACE_SHA256)
    );
    assert_eq!(
        fixture["gf2_helper_sha256"].as_str(),
        Some(PYTHON_GF2_SHA256)
    );
    assert_eq!(
        fixture["feature_packet_sha256"].as_str(),
        Some(PYTHON_PACKET_SHA256)
    );
    assert_eq!(
        fixture["feature_ids"],
        serde_json::json!(["C01", "D03", "H01", "H02", "H03", "H04"])
    );

    for batch in fixture["batches"]
        .as_array()
        .expect("fixture must contain width-specific batches")
    {
        let states = states_from_batch(batch, true);
        assert_feature_rows(
            batch["features"].as_array().unwrap(),
            &states,
            "python-v17-subspace-parity-v1",
            "exact_relabel",
            false,
        );
    }
}

#[test]
fn all_python_v17_basis_gradient_features_match_rust_on_widths_four_to_six() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/python_basis_gradient_3.json"))
            .expect("pinned Python basis-gradient fixture must be valid JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("python_basis_gradient_reference_v1")
    );
    assert_eq!(
        fixture["source_archive_sha256"].as_str(),
        Some(PYTHON_ARCHIVE_SHA256)
    );
    assert_eq!(
        fixture["basis_gradient_sha256"].as_str(),
        Some(PYTHON_BASIS_GRADIENT_SHA256)
    );
    assert_eq!(
        fixture["basis_macros_sha256"].as_str(),
        Some(PYTHON_BASIS_MACROS_SHA256)
    );
    assert_eq!(
        fixture["gf2_helper_sha256"].as_str(),
        Some(PYTHON_GF2_SHA256)
    );
    assert_eq!(
        fixture["feature_packet_sha256"].as_str(),
        Some(PYTHON_PACKET_SHA256)
    );
    assert_eq!(
        fixture["feature_ids"],
        serde_json::json!(["E01", "E04", "E05"])
    );

    let batches = fixture["batches"]
        .as_array()
        .expect("fixture must contain width-specific batches");
    assert_eq!(batches.len(), 3);
    for (batch, expected_width) in batches.iter().zip([4u64, 5, 6]) {
        assert_eq!(batch["n"].as_u64(), Some(expected_width));
        assert_eq!(batch["state_count"].as_u64(), Some(32));
        let states = states_from_batch(batch, false);
        assert_feature_rows(
            batch["features"].as_array().unwrap(),
            &states,
            "python-v17-basis-gradient-parity-v1",
            "augmentation_only",
            true,
        );
    }
}

fn assert_diffusion_feature_rows(
    rows: &[Value],
    states: &[MatrixState],
    seed: u64,
    symmetry_contract: &str,
) {
    let ids = rows
        .iter()
        .map(|row| parse_id(&row["id"]))
        .collect::<Vec<_>>();
    let policy = FeaturePolicy {
        enabled_ids: ids,
        seed,
        schema_id: "python-v17-diffusion-parity-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes: None,
    };
    let mut registry = FeatureRegistry::new();
    register_diffusion(
        &mut registry,
        DiffusionConfig {
            taus: [0.0, 0.25, 1.0, 4.0],
            floor: 1e-30,
            max_n: 16,
            max_pair_states: 10_000,
            mc_samples: 128,
            mc_tau: 1.0,
        },
    )
    .expect("source diffusion providers");
    registry
        .selected_specs(&policy)
        .expect("every pinned diffusion provider must be executable");
    let packet = registry
        .build(states, &policy)
        .expect("Rust must build every pinned diffusion feature");

    for expected in rows {
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
            let tolerance = 3e-6 * (1.0 + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "{id}[{index}]: Rust={actual:.9}, Python={expected:.9}, tolerance={tolerance:.3e}"
            );
        }
    }
}

#[test]
fn all_python_v17_diffusion_features_match_rust_on_pinned_states() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/python_diffusion_10.json"))
        .expect("pinned Python diffusion fixture must be valid JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("python_diffusion_reference_v1")
    );
    assert_eq!(
        fixture["source_archive_sha256"].as_str(),
        Some(PYTHON_ARCHIVE_SHA256)
    );
    assert_eq!(
        fixture["diffusion_module_sha256"].as_str(),
        Some("c618292a54d10a092abc44cd808b892b51b73e5409633434be201fe5599fcf01")
    );
    assert_eq!(
        fixture["probes_module_sha256"].as_str(),
        Some("183a37e9325c53b16c9c951e40e9cbb0fcb0bdc0c0bfb7963af977e14c70b4e4")
    );
    assert_eq!(fixture["numpy_version"].as_str(), Some("2.2.6"));
    assert_eq!(fixture["scipy_version"].as_str(), Some("1.15.3"));
    assert_eq!(
        fixture["config"]["taus"],
        serde_json::json!([0.0, 0.25, 1.0, 4.0])
    );
    assert_eq!(fixture["config"]["floor"].as_f64(), Some(1e-30));
    assert_eq!(fixture["config"]["max_n"].as_u64(), Some(16));
    assert_eq!(fixture["config"]["max_pair_states"].as_u64(), Some(10_000));
    assert_eq!(fixture["config"]["mc_samples"].as_u64(), Some(128));
    assert_eq!(fixture["config"]["mc_tau"].as_f64(), Some(1.0));
    let expected_ids = (1..=10)
        .map(|index| format!("G{index:02}"))
        .collect::<Vec<_>>();
    assert_eq!(fixture["feature_ids"], serde_json::json!(expected_ids));
    assert_eq!(fixture["mc_seed"].as_u64(), Some(91));

    let batches = fixture["batches"]
        .as_array()
        .expect("fixture must contain width-specific batches");
    assert_eq!(batches.len(), 5);
    for (batch, expected_width) in batches.iter().zip(2u64..=6) {
        assert_eq!(batch["n"].as_u64(), Some(expected_width));
        let expected_state_count = match expected_width {
            2 => 6,
            3 => 168,
            _ => 12,
        };
        assert_eq!(batch["state_count"].as_u64(), Some(expected_state_count));
        let exact_rows = batch["exact_features"].as_array().unwrap();
        let exact_ids = exact_rows
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(exact_ids, ["G01", "G02", "G03", "G04", "G05", "G06"]);
        let monte_carlo_rows = batch["monte_carlo_features"].as_array().unwrap();
        let monte_carlo_ids = monte_carlo_rows
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(monte_carlo_ids, ["G07", "G08", "G09", "G10"]);
        let states = states_from_batch(batch, false);
        assert_eq!(batch["state_count"].as_u64(), Some(states.len() as u64));
        assert_diffusion_feature_rows(
            batch["exact_features"].as_array().unwrap(),
            &states,
            0,
            "exact_relabel",
        );
        assert_diffusion_feature_rows(
            batch["monte_carlo_features"].as_array().unwrap(),
            &states,
            91,
            "augmentation_only",
        );
    }
}
