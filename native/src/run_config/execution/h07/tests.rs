//! Independent scalar-built packets, host feature statuses and the original
//! compact-packet field evaluate the exact same checkpoint and physical probes.
use super::*;
use crate::view_diagnostic::resident::tests::word_views;
use crate::{compute_h07_prediction_spread, evaluate_h07_prediction_spread, H07ModelProbe};
use std::path::PathBuf;

#[test]
#[ignore = "requires pinned H07 fixture and a real trained model"]
fn original_compact_field_matches_complete_cuda_h07_on_frozen_dev() {
    let path = PathBuf::from(std::env::var_os("CNOT103_H07_ORACLE_INPUT").unwrap());
    let spec: H07Manifest = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let DevCheckpoint { field, receipt, .. } = load_dev_checkpoint(
        &path,
        &spec.training_manifest,
        &spec.training_result,
        &spec.dev_indices,
    )
    .unwrap();
    let file: Value =
        serde_json::from_slice(&fs::read(path.parent().unwrap().join("result.json")).unwrap())
            .unwrap();
    let binding = model_binding(
        &spec.training_manifest.sha256,
        &receipt,
        &field.registry.contract_sha256(),
    );
    let model_hash = artifacts::sha256(&serde_json::to_vec(&binding).unwrap());
    let mut max_prediction_error = 0f64;
    let mut max_variance_error = 0f64;
    let mut statuses = 0;
    let mut scopes = std::collections::HashSet::new();
    let mut digests = std::collections::HashSet::new();
    for (&i, file_row) in spec
        .dev_indices
        .iter()
        .zip(file["rows"].as_array().unwrap())
    {
        let record = &field.dataset.dev.records()[i];
        let root = &record.state;
        let alphabet = field.objective.alphabet_for(root.alphabet_id()).unwrap();
        let (scope, actual) = resident::evaluate(
            root,
            alphabet,
            &field.model,
            &field.plans[&root.dimension()],
            &model_hash,
        )
        .unwrap();
        assert_eq!(file_row["dev_index"], i);
        assert_eq!(file_row["problem_scope"], scope);
        let formatted = resident::report_json(&actual);
        // JSON parsing may round decimal f64 text by one ulp. The digest binds
        // the complete in-memory bits and every individual input audit.
        assert_eq!(
            file_row["diagnostic"]["probe_set_sha256"],
            formatted["probe_set_sha256"]
        );
        assert_eq!(file_row["diagnostic"]["model_contract_sha256"], model_hash);
        assert!(scopes.insert(actual.problem_scope_sha256.clone()));
        assert!(digests.insert(actual.probe_set_sha256.clone()));
        let construction: Vec<_> = record.verified_words[0]
            .gates
            .iter()
            .rev()
            .copied()
            .collect();
        let rows = word_views(root.dimension(), &construction);
        assert_eq!(rows[0].2, *root);
        let mut packets = Vec::new();
        for (_, _, row) in &rows {
            assert!(field.dataset.forbidden_keys.contains(row));
            packets.push(
                field
                    .registry
                    .build(std::slice::from_ref(row), field.objective.feature_policy())
                    .unwrap(),
            );
        }
        let probes: Vec<_> = rows
            .iter()
            .zip(&packets)
            .map(|((id, kind, _), packet)| H07ModelProbe {
                probe_id: (*id).into(),
                kind: *kind,
                packet,
            })
            .collect();
        let expected = evaluate_h07_prediction_spread(
            &field.model,
            &model_hash,
            &actual.problem_scope_sha256,
            &probes,
        )
        .unwrap();
        assert_eq!(actual.probes.len(), expected.probes.len());
        for (p, q) in actual.probes.iter().zip(&expected.probes) {
            assert_eq!((&p.probe_id, p.kind), (&q.probe_id, q.kind));
            assert_eq!(p.input_coverage, q.input_coverage);
            assert!(p.input_coverage.as_ref().unwrap().is_complete());
            statuses += p.input_coverage.as_ref().unwrap().entries.len();
            let err = (p.prediction - q.prediction).abs();
            max_prediction_error = max_prediction_error.max(err);
            assert!(
                err <= 2e-5 * (1. + q.prediction.abs()),
                "dev[{i}] {}: {err}",
                p.probe_id
            );
        }
        max_variance_error = max_variance_error
            .max((actual.population_variance - expected.population_variance).abs());
        assert!((actual.population_variance - expected.population_variance).abs() <= 2e-7);
        // Compare device reduction and exact audit/digest against scalar v3
        // applied to the resident model's actual FP32 predictions.
        assert_eq!(
            actual,
            compute_h07_prediction_spread(
                &model_hash,
                &actual.problem_scope_sha256,
                &actual.probes
            )
            .unwrap()
        );
    }
    println!("H07 verified {} dev roots x 5 physical probe forwards, {statuses} source status/reason rows; max value error {max_prediction_error:e}, max variance error {max_variance_error:e}",spec.dev_indices.len());

    // These hand-built states exercise algebraic and failure branches only.
    // They are separate from the sealed holdout report.
    let base = &field.dataset.dev.records()[0].state;
    let alphabet = field.objective.alphabet_for(base.alphabet_id()).unwrap();
    let plan = &field.plans[&4];
    let gate = |control, target| crate::CnotGate { control, target };
    let transformations = word_views(4, &[gate(0, 1), gate(1, 2)]);
    let (_, physical) = resident::evaluate(
        &transformations[0].2,
        alphabet,
        &field.model,
        plan,
        &model_hash,
    )
    .unwrap();
    assert_ne!(transformations[0].2, transformations[1].2);
    let mut inverse_predictions = Vec::new();
    for ((id, _, state), prediction) in transformations.iter().zip(&physical.probes) {
        let packet = field
            .registry
            .build(
                std::slice::from_ref(state),
                field.objective.feature_policy(),
            )
            .unwrap();
        let v = tch::no_grad(|| field.model.forward(&packet))
            .unwrap()
            .value
            .view([-1])
            .double_value(&[0]);
        assert_eq!(*id, prediction.probe_id);
        assert!((prediction.prediction - v).abs() <= 2e-5 * (1. + v.abs()));
        inverse_predictions.push(prediction.prediction);
    }
    assert_ne!(inverse_predictions[0], inverse_predictions[1]);
    let identity = crate::MatrixState::identity(4, alphabet.id()).unwrap();
    let (_, terminal) =
        resident::evaluate(&identity, alphabet, &field.model, plan, &model_hash).unwrap();
    assert_eq!(
        (
            terminal.mean,
            terminal.population_variance,
            terminal.channel
        ),
        (0., 0., 0.)
    );
    let outside =
        crate::MatrixState::from_packed_rows(4, alphabet.id(), vec![3, 2, 6, 12]).unwrap();
    let err = resident::evaluate(&outside, alphabet, &field.model, plan, &model_hash).unwrap_err();
    assert!(
        err.contains("canonical") && err.contains("F02 unsupported"),
        "{err}"
    );
    println!("trained H07 extra controls: distinct canonical/inverse values {inverse_predictions:?}; identity zero; hard coverage error: {err}");
}
