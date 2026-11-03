#![cfg(feature = "torch")]

use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    CnotAlphabet, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole, FeatureStatus,
    MatrixState,
};
use serde_json::Value;
use std::collections::HashSet;
use tch::nn::VarStore;
use tch::{Device, Tensor};

const PYTHON_ARCHIVE_SHA256: &str =
    "9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866";
const SCHEMA_ID: &str = "python-forward-parity-structural-33-v1";

fn number_array(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("fixture tensor values must be an array")
        .iter()
        .map(|number| number.as_f64().expect("fixture value must be numeric") as f32)
        .collect()
}

fn parse_id(value: &Value) -> FeatureId {
    let text = value.as_str().expect("feature ID must be text");
    let bytes = text.as_bytes();
    assert!(bytes.len() >= 2, "invalid feature ID: {text}");
    let ordinal = text[1..]
        .parse::<u8>()
        .expect("feature ordinal must be numeric");
    FeatureId::new(bytes[0], ordinal).expect("feature ID must be canonical")
}

fn parse_role(value: &Value) -> FeatureRole {
    match value.as_str().expect("feature role must be text") {
        "node" => FeatureRole::Node,
        "pair" => FeatureRole::Pair,
        "global" => FeatureRole::Global,
        "group" => FeatureRole::Group,
        unknown => panic!("unknown feature role: {unknown}"),
    }
}

fn deterministic_tensor_values(tensor_index: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|flat_index| {
            let numerator = ((tensor_index * 131 + flat_index * 17 + 43) % 257) as i32 - 128;
            numerator as f32 / 512.0
        })
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
fn native_forward_matches_pinned_python_reference_with_all_active_structural_features() {
    tch::set_num_threads(1);
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/python_structural_33_forward_small.json"
    ))
    .expect("pinned all-structural Python forward fixture must be valid JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("python_reference_forward_structural_33_v1")
    );
    assert_eq!(
        fixture["reference_archive_sha256"].as_str(),
        Some(PYTHON_ARCHIVE_SHA256)
    );
    assert_eq!(
        fixture["weight_scheme"].as_str(),
        Some("sorted_state_dict_mod257_v1")
    );
    let feature_ids = fixture["feature_ids"]
        .as_array()
        .expect("feature IDs must be an array")
        .iter()
        .map(parse_id)
        .collect::<Vec<_>>();
    assert_eq!(feature_ids.len(), 33);

    let state_rows = fixture["states"]
        .as_array()
        .expect("states must be an array");
    let states = state_rows
        .iter()
        .map(|row| {
            let n = row["n"].as_u64().expect("state width must be numeric") as u32;
            let alphabet = CnotAlphabet::all_to_all_unit(n).expect("valid CNOT alphabet");
            let packed_rows = row["packed_rows"]
                .as_array()
                .expect("packed rows must be an array")
                .iter()
                .map(|value| value.as_u64().expect("packed row must be unsigned"))
                .collect::<Vec<_>>();
            MatrixState::from_packed_rows(n, alphabet.id(), packed_rows)
                .expect("fixture state must belong to GL(n,2)")
        })
        .collect::<Vec<_>>();
    assert_eq!(states.len(), 4);
    assert_eq!(states[0].dimension(), 3);
    assert_eq!(states[2].dimension(), 4);

    let policy = FeaturePolicy {
        enabled_ids: feature_ids,
        seed: 0,
        schema_id: SCHEMA_ID.into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let registry = FeatureRegistry::new();
    let packet = registry
        .build(&states, &policy)
        .expect("Rust must build all 33 source-backed structural features");
    assert_eq!(
        packet.entries.len(),
        103,
        "Rust packet preserves the canonical feature manifest"
    );

    let expected_features = fixture["feature_values"]
        .as_array()
        .expect("feature references must be an array");
    assert_eq!(expected_features.len(), 33);
    for expected in expected_features {
        let id = parse_id(&expected["id"]);
        let actual = packet.feature(id);
        assert_eq!(actual.spec.role, parse_role(&expected["role"]), "{id}");
        assert_eq!(
            actual.spec.channels,
            expected["channels"].as_u64().unwrap() as usize,
            "{id} channels"
        );
        let shape = expected["shape"]
            .as_array()
            .expect("feature shape must be an array")
            .iter()
            .map(|extent| extent.as_u64().expect("shape extent must be unsigned") as usize)
            .collect::<Vec<_>>();
        assert_eq!(actual.shape, shape, "{id} shape");
        assert!(
            actual.available.iter().all(|available| *available == 1),
            "{id} source feature must be available for every fixture state"
        );
        assert!(
            actual
                .status
                .iter()
                .all(|status| *status == FeatureStatus::Computed),
            "{id} status"
        );
        let expected_values = number_array(&expected["values"]);
        assert_eq!(actual.values.len(), expected_values.len(), "{id} values");
        for (index, (actual, expected)) in actual.values.iter().zip(expected_values).enumerate() {
            assert_close(
                &format!("{id} source feature[{index}]"),
                f64::from(*actual),
                f64::from(expected),
            );
        }
    }

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

    let variables = VarStore::new(Device::Cpu);
    let model = FusionDistanceField::new_for_policy(
        &variables.root(),
        &registry,
        &policy,
        Device::Cpu,
        config,
    )
    .expect("native model must accept the 33-feature source profile");

    let mut actual_variables = variables.variables().into_iter().collect::<Vec<_>>();
    actual_variables.sort_by(|left, right| left.0.cmp(&right.0));
    let reference_shapes = fixture["state_dict_shapes"]
        .as_object()
        .expect("Python state-dict shapes must be an object");
    assert_eq!(actual_variables.len(), reference_shapes.len());
    assert_eq!(
        actual_variables
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        reference_shapes
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        "Python and Rust parameter names differ"
    );

    for (tensor_index, (name, target)) in actual_variables.iter().enumerate() {
        let shape = reference_shapes[name]
            .as_array()
            .expect("parameter shape must be an array")
            .iter()
            .map(|extent| extent.as_i64().expect("parameter extent must be signed"))
            .collect::<Vec<_>>();
        assert_eq!(target.size(), shape, "parameter shape differs: {name}");
        let values = deterministic_tensor_values(tensor_index, target.numel());
        let source = Tensor::from_slice(&values).view(shape.as_slice());
        let mut destination = target.shallow_clone();
        tch::no_grad(|| destination.copy_(&source));
    }

    let output = model.forward(&packet).expect("native forward must succeed");
    let expected_values = number_array(&fixture["expected"]["value"]);
    assert_eq!(expected_values.len(), states.len());
    for (row, expected) in expected_values.iter().enumerate() {
        let actual = output.value.double_value(&[row as i64]);
        if states[row].is_identity() {
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
        for control in 0..packet.nodes as i64 {
            for target in 0..packet.nodes as i64 {
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
