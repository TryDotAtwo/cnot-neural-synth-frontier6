#![cfg(feature = "torch")]
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, MatrixState,
};
use serde_json::Value;
use std::collections::HashSet;
use tch::nn::VarStore;
use tch::{Device, Tensor};

const PYTHON_ARCHIVE_SHA256: &str =
    "9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866";

fn number_array(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("fixture tensor values must be an array")
        .iter()
        .map(|number| number.as_f64().expect("fixture value must be numeric") as f32)
        .collect()
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    let tolerance = 2e-5 * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: Rust={actual:.9}, Python={expected:.9}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn native_forward_matches_pinned_python_reference_with_identical_weights() {
    tch::set_num_threads(1);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/python_a01_forward_small.json"))
            .expect("pinned Python forward fixture must be valid JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("python_reference_forward_v1")
    );
    assert_eq!(
        fixture["reference_archive_sha256"].as_str(),
        Some(PYTHON_ARCHIVE_SHA256)
    );
    assert_eq!(fixture["feature_ids"], serde_json::json!(["A01"]));

    let config = ModelConfig {
        wire_dim: 8,
        pair_dim: 4,
        structural_blocks: 1,
        attention_heads: 2,
        feedforward_dim: 16,
        fusion_blocks: 1,
        basis_proposal_hidden_dim: 128,
    };
    assert_eq!(
        fixture["config"]["basis_proposal_hidden_dim"].as_u64(),
        Some(config.basis_proposal_hidden_dim as u64)
    );

    let alphabet = CnotAlphabet::all_to_all_unit(3).expect("valid width-3 alphabet");
    let mut nonidentity = MatrixState::identity(3, alphabet.id()).expect("identity state");
    nonidentity
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .expect("first CNOT");
    nonidentity
        .apply_cnot(CnotGate {
            control: 1,
            target: 2,
        })
        .expect("second CNOT");
    let states = vec![
        MatrixState::identity(3, alphabet.id()).expect("identity state"),
        nonidentity,
    ];

    let id = FeatureId::new(b'A', 1).expect("A01 is in the feature manifest");
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "python-forward-parity-a01-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let registry = FeatureRegistry::new();
    let packet = registry
        .build(&states, &policy)
        .expect("the source-backed A01 producer must build");
    let expected_features = number_array(&fixture["feature_values"]);
    assert_eq!(packet.feature(id).values.len(), expected_features.len());
    for (index, (actual, expected)) in packet
        .feature(id)
        .values
        .iter()
        .zip(expected_features)
        .enumerate()
    {
        assert_close(
            &format!("A01 feature[{index}]"),
            f64::from(*actual),
            f64::from(expected),
        );
    }

    let variables = VarStore::new(Device::Cpu);
    let model = FusionDistanceField::new_for_policy(
        &variables.root(),
        &registry,
        &policy,
        Device::Cpu,
        config,
    )
    .expect("native model must accept the same A01 profile");
    let actual_variables = variables.variables();
    let reference_state = fixture["state_dict"]
        .as_object()
        .expect("reference state_dict must be an object");
    assert_eq!(
        actual_variables.len(),
        reference_state.len(),
        "Python and Rust must expose the same parameter set"
    );
    for (name, reference) in reference_state {
        let target = actual_variables
            .get(name)
            .unwrap_or_else(|| panic!("native parameter is missing: {name}"));
        let shape = reference["shape"]
            .as_array()
            .expect("reference shape must be an array")
            .iter()
            .map(|extent| extent.as_i64().expect("shape extent must be an integer"))
            .collect::<Vec<_>>();
        assert_eq!(target.size(), shape, "parameter shape differs: {name}");
        let values = number_array(&reference["values"]);
        let source = Tensor::from_slice(&values).view(shape.as_slice());
        let mut destination = target.shallow_clone();
        tch::no_grad(|| destination.copy_(&source));
    }

    let output = model.forward(&packet).expect("native forward must succeed");
    let expected_values = number_array(&fixture["expected"]["value"]);
    assert_eq!(expected_values.len(), states.len());
    for (row, expected) in expected_values.iter().enumerate() {
        let actual = output.value.double_value(&[row as i64]);
        if row == 0 {
            assert_eq!(actual, 0.0, "identity distance is an exact boundary");
        } else {
            assert_close("value", actual, f64::from(*expected));
        }
    }

    let expected_actions = fixture["expected"]["allowed_actions"]
        .as_array()
        .expect("allowed actions must be an array")
        .iter()
        .map(|action| {
            let indices = action
                .as_array()
                .expect("action coordinates must be an array")
                .iter()
                .map(|coordinate| coordinate.as_u64().expect("coordinate must be unsigned"))
                .collect::<Vec<_>>();
            assert_eq!(indices.len(), 3);
            [indices[0] as i64, indices[1] as i64, indices[2] as i64]
        })
        .collect::<Vec<_>>();
    let expected_allowed = expected_actions.iter().copied().collect::<HashSet<_>>();
    assert_eq!(expected_allowed.len(), expected_actions.len());
    let expected_proposal = number_array(&fixture["expected"]["proposal"]);
    let expected_qhat = number_array(&fixture["expected"]["qhat"]);
    assert_eq!(expected_actions.len(), expected_proposal.len());
    assert_eq!(expected_actions.len(), expected_qhat.len());

    for batch in 0..states.len() as i64 {
        for control in 0..3i64 {
            for target in 0..3i64 {
                let expected = if expected_allowed.contains(&[batch, control, target]) {
                    1.0
                } else {
                    0.0
                };
                assert_close(
                    "full action mask",
                    output.action_mask.double_value(&[batch, control, target]),
                    expected,
                );
            }
        }
    }

    for (index, [batch, control, target]) in expected_actions.iter().copied().enumerate() {
        assert_close(
            "proposal logit",
            output
                .proposal_logits
                .double_value(&[batch, control, target]),
            f64::from(expected_proposal[index]),
        );
        assert_close(
            "qhat",
            output.qhat.double_value(&[batch, control, target]),
            f64::from(expected_qhat[index]),
        );
    }
}
