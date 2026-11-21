//! The scalar registry and original compact-packet forward are independent
//! oracles. Their outputs are read only in this test.
use super::*;
use crate::run_config::execution::field::BoundField;
use crate::{device_state::DeviceStateBatch, CnotAlphabet, FeatureStatus, MatrixState};
use std::path::PathBuf;
use tch::{Device, Tensor};

fn reference(field: &BoundField, states: &[MatrixState]) -> crate::model::FieldOutput {
    let packet = field
        .registry
        .build(states, field.objective.feature_policy())
        .unwrap();
    tch::no_grad(|| field.model.forward(&packet)).unwrap()
}

fn check_word(
    field: &BoundField,
    root: &MatrixState,
    a: &CnotAlphabet,
    head: ActionHead,
    budget: BeamConfig,
    report: &Value,
) -> f64 {
    assert_eq!(
        (budget.max_depth, budget.beam_width, budget.shortlist),
        (2, 11, 11)
    );
    let output = reference(field, std::slice::from_ref(root));
    let q = match head {
        ActionHead::Qhat => output.qhat,
        ActionHead::Proposal => -output.proposal_logits,
    };
    let mut ids: Vec<_> = (0..a.gates().len()).collect();
    let qscore = |i: usize| {
        let g = a.gates()[i];
        q.double_value(&[0, i64::from(g.control), i64::from(g.target)])
    };
    ids.sort_by(|&i, &j| qscore(i).total_cmp(&qscore(j)).then(i.cmp(&j)));
    ids.truncate(budget.shortlist as usize);
    let mut candidates = Vec::new();
    for indices in ids.chunks(budget.inference_batch as usize) {
        let states: Vec<_> = indices
            .iter()
            .map(|&i| {
                let mut s = root.clone();
                s.apply_cnot(a.gates()[i]).unwrap();
                s
            })
            .collect();
        let v = reference(field, &states).value.view([-1]);
        for (j, (&first, s)) in indices.iter().zip(states).enumerate() {
            let score = v.double_value(&[j as i64]) + 1.0;
            assert!(score.is_finite());
            candidates.push((score, first, s));
        }
    }
    candidates.sort_by(|l, r| l.0.total_cmp(&r.0).then(l.1.cmp(&r.1)));
    let margin = candidates.last().unwrap().0 - candidates.first().unwrap().0;
    let mut hits = Vec::new();
    for (_, first, s) in candidates.into_iter().take(budget.beam_width as usize) {
        for (i, &g) in a.gates().iter().enumerate() {
            let mut x = s.clone();
            x.apply_cnot(g).unwrap();
            if x.is_identity() {
                hits.push([first, i]);
            }
        }
    }
    let chosen = hits
        .into_iter()
        .min()
        .expect("the complete final goal check found a word");
    let expected = chosen.map(|i| [a.gates()[i].control, a.gates()[i].target]);
    assert_eq!(report["word"]["control_target"], json!(expected));
    assert_eq!(report["expanded"], 12);
    assert_eq!(report["generated"], 11);
    assert_eq!(report["value_evaluations"], 11);
    margin
}

#[test]
#[ignore = "requires the trained GL4 dev-search fixture and its two independent CLI reports"]
fn compare_dev_neighborhood_coverage_with_source_providers() {
    let path = PathBuf::from(
        std::env::var_os("CNOT103_SEARCH_ORACLE_INPUT").expect("fixture search manifest"),
    );
    let search: SearchManifest = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let PanelCheckpoint {
        field,
        receipt,
        device,
        ..
    } = load_panel_checkpoint(
        &path,
        &search.training_manifest,
        &search.training_result,
        &search.panel,
        &search.panel_indices,
    )
    .unwrap();
    let mut unavailable = std::collections::BTreeMap::<String, usize>::new();
    let mut rows = 0;
    let mut max_value_diff = 0.0_f64;
    let mut max_action_diff = 0.0_f64;
    let n = 4i64;
    let alphabet = CnotAlphabet::all_to_all_unit(n as u32).unwrap();
    let positions = Tensor::from_slice(
        &alphabet
            .gates()
            .iter()
            .map(|g| i64::from(g.control) * n + i64::from(g.target))
            .collect::<Vec<_>>(),
    )
    .to_device(Device::Cuda(device));
    for (i, record) in field.dataset.dev.records().iter().enumerate() {
        let mut states = vec![record.state.clone()];
        states.extend(alphabet.gates().iter().map(|g| {
            let mut s = record.state.clone();
            s.apply_cnot(*g).unwrap();
            s
        }));
        for chunk in states.chunks(4) {
            let source = field
                .registry
                .build(chunk, field.objective.feature_policy())
                .unwrap();
            let packed: Vec<_> = chunk
                .iter()
                .flat_map(MatrixState::packed_rows)
                .map(|v| *v as i64)
                .collect();
            let t = Tensor::from_slice(&packed)
                .view([chunk.len() as i64, n, 1])
                .to_device(Device::Cuda(device));
            let packet = field.plans[&(n as u32)]
                .build(
                    &DeviceStateBatch::from_packed_tensor_for_alphabet(t, &alphabet, device)
                        .unwrap(),
                )
                .unwrap();
            let actual = tch::no_grad(|| field.model.forward_device(&packet)).unwrap();
            let expected = tch::no_grad(|| field.model.forward(&source)).unwrap();
            let diff = (&actual.value - &expected.value)
                .abs()
                .max()
                .double_value(&[]);
            max_value_diff = max_value_diff.max(diff);
            assert!(diff < 2e-5);
            for (lhs, rhs) in [
                (actual.qhat, expected.qhat),
                (actual.proposal_logits, expected.proposal_logits),
            ] {
                let diff = (lhs.flatten(1, -1).index_select(1, &positions)
                    - rhs.flatten(1, -1).index_select(1, &positions))
                .abs()
                .max()
                .double_value(&[]);
                max_action_diff = max_action_diff.max(diff);
                assert!(diff < 2e-5);
            }
            for e in source.entries {
                if e.spec.id == crate::schema::H07_POST_FORWARD_DIAGNOSTIC_ID {
                    continue;
                }
                let statuses: Vec<u8> = Vec::try_from(
                    packet
                        .feature(e.spec.id)
                        .unwrap()
                        .status()
                        .to_device(Device::Cpu),
                )
                .unwrap();
                for (j, status) in e.status.into_iter().enumerate() {
                    let code = match status {
                        FeatureStatus::Computed => 1,
                        FeatureStatus::Inapplicable => 2,
                        FeatureStatus::Unsupported => 3,
                        FeatureStatus::ResourceLimit => 4,
                        FeatureStatus::NotRequested => 0,
                    };
                    assert_eq!(
                        statuses[j],
                        code,
                        "{} on {:?}",
                        e.spec.id,
                        chunk[j].packed_rows()
                    );
                    if !matches!(code, 1 | 2) {
                        *unavailable
                            .entry(format!(
                                "dev[{i}] {} {} {}",
                                e.spec.id,
                                status.as_str(),
                                e.reason[j]
                            ))
                            .or_default() += 1;
                    }
                }
            }
            rows += chunk.len();
        }
    }
    for (name, head) in [
        ("qhat", ActionHead::Qhat),
        ("proposal", ActionHead::Proposal),
    ] {
        let report: Value = serde_json::from_slice(
            &fs::read(path.with_file_name(format!("result-{name}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(report["checkpoint"]["sha256"], receipt.sha256);
        for row in report["rows"].as_array().unwrap() {
            let i = row["dev_index"].as_u64().unwrap() as usize;
            let margin = check_word(
                &field,
                &field.dataset.dev.records()[i].state,
                &alphabet,
                head,
                search.search,
                row,
            );
            println!("source compact-packet model and independent {name} beam reproduce dev[{i}] with V range {margin}");
        }
    }
    println!("source/device coverage and heads on {rows} roots/children x 102 inputs; max V={max_value_diff} max Q/proposal={max_action_diff}; strict exceptions: {unavailable:#?}");
}
