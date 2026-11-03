use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use cnot103_native::run_config::parse_run_manifest;
use cnot103_native::shear::partial_shear;
use cnot103_native::{CnotAlphabet, CnotGate, MatrixState};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const TRAINER: &str = env!("CARGO_BIN_EXE_cnot103-train");
const DIMENSION: u32 = 4;

fn all_paths(alphabet: &CnotAlphabet) -> HashMap<MatrixState, Vec<CnotGate>> {
    let identity = MatrixState::identity(DIMENSION, alphabet.id()).unwrap();
    let mut paths = HashMap::from([(identity.clone(), Vec::new())]);
    let mut queue = VecDeque::from([identity]);
    while let Some(state) = queue.pop_front() {
        let path = paths.get(&state).unwrap().clone();
        for gate in alphabet.gates() {
            let mut next = state.clone();
            next.apply_cnot(*gate).unwrap();
            if !paths.contains_key(&next) {
                let mut next_path = path.clone();
                next_path.push(*gate);
                paths.insert(next.clone(), next_path);
                queue.push_back(next);
            }
        }
    }
    paths
}

fn apply_path(mut state: MatrixState, gates: &[CnotGate]) -> MatrixState {
    for gate in gates {
        state.apply_cnot(*gate).unwrap();
    }
    state
}

fn transpose(state: &MatrixState) -> MatrixState {
    let n = state.dimension() as usize;
    let mut rows = vec![0u64; n];
    for (row, packed_row) in state.packed_rows().iter().enumerate() {
        for (column, transposed_row) in rows.iter_mut().enumerate() {
            if ((packed_row >> column) & 1) != 0 {
                *transposed_row |= 1u64 << row;
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), rows).unwrap()
}

fn reverse_wires(state: &MatrixState) -> MatrixState {
    let n = state.dimension() as usize;
    let mut rows = vec![0u64; n];
    for (packed_row, reversed_row) in state.packed_rows().iter().zip(rows.iter_mut().rev()) {
        for old_column in 0..n {
            if ((packed_row >> old_column) & 1) != 0 {
                *reversed_row |= 1u64 << (n - 1 - old_column);
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), rows).unwrap()
}

fn finite_views(
    state: &MatrixState,
    paths: &HashMap<MatrixState, Vec<CnotGate>>,
) -> HashSet<MatrixState> {
    let inverse_path = paths
        .get(state)
        .unwrap()
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let inverse = apply_path(
        MatrixState::identity(DIMENSION, state.alphabet_id()).unwrap(),
        &inverse_path,
    );
    let transposed = transpose(state);
    let inverse_transposed = transpose(&inverse);
    let mut views = HashSet::new();
    for matrix in [state.clone(), inverse, transposed, inverse_transposed] {
        views.insert(matrix.clone());
        views.insert(reverse_wires(&matrix));
    }
    views
}

fn record_json(
    state: &MatrixState,
    paths: &HashMap<MatrixState, Vec<CnotGate>>,
    split: &str,
) -> Value {
    let reduction = paths
        .get(state)
        .unwrap()
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let identity = MatrixState::identity(DIMENSION, state.alphabet_id()).unwrap();
    let lineage = format!("fixture-{split}");
    json!({
        "state": {"n": DIMENSION, "alphabet_id": state.alphabet_id(), "packed_rows": state.packed_rows()},
        "verified_words": [{
            "start": {"n": DIMENSION, "alphabet_id": state.alphabet_id(), "packed_rows": state.packed_rows()},
            "end": {"n": DIMENSION, "alphabet_id": identity.alphabet_id(), "packed_rows": identity.packed_rows()},
            "gates": reduction.iter().map(|gate| json!({"control": gate.control, "target": gate.target})).collect::<Vec<_>>(),
            "physical_cost": reduction.len(), "producer": "integration-test-bfs-gl4", "lineage_id": lineage
        }],
        "lineage_id": lineage,
        "seed": match split { "train" => 1, "dev" => 2, _ => 3 },
        "source": format!("integration-{split}"), "split": split,
        "exact_distance": reduction.len(), "exact_evidence": "exhaustive GL(4,2) BFS fixture"
    })
}

fn write_pinned_json(directory: &Path, filename: &str, body: &Value) -> String {
    let path = directory.join(filename);
    let bytes = serde_json::to_vec(body).unwrap();
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    fs::write(&path, &bytes).unwrap();
    fs::write(path.with_extension("sha256"), format!("{digest}\n")).unwrap();
    digest
}

fn run_cli(manifest_path: &Path) -> Value {
    let output = Command::new(TRAINER)
        .arg(manifest_path)
        .output()
        .expect("spawn cnot103-train in a fresh process");
    assert!(
        output.status.success(),
        "cnot103-train failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("CLI prints a JSON result")
}

fn has_supported_partial_shear(state: &MatrixState) -> bool {
    matches!(partial_shear(state), Ok(Some(d))
        if d.source_wires.len() >= 2 && d.block.iter().flatten().any(|&bit| bit != 0))
}

#[test]
fn run_manifest_rejects_wrong_schema_before_loading_artifacts() {
    let error = parse_run_manifest(br#"{"schema":"not-a-training-run"}"#)
        .expect_err("wrong schema must fail before artifact I/O");
    assert!(error.to_string().contains("schema"));
}

#[test]
fn run_manifest_rejects_unknown_top_level_fields() {
    let error =
        parse_run_manifest(br#"{"schema":"cnot103_training_run_v1","invented_default":true}"#)
            .expect_err("typos and unknown controls must not be silently ignored");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn run_manifest_rejects_invalid_json() {
    assert!(parse_run_manifest(b"{").is_err());
}

#[test]
#[ignore = "runs the full 102-input CLI step/save/fresh-process-resume workflow on CUDA"]
fn cli_fresh_checkpoint_resume_executes_next_update_in_a_new_process() {
    let alphabet = CnotAlphabet::all_to_all_unit(DIMENSION).unwrap();
    let paths = all_paths(&alphabet);
    assert_eq!(paths.len(), 20_160, "fixture must cover all GL(4,2)");

    let identity = MatrixState::identity(DIMENSION, alphabet.id()).unwrap();
    let mut dev = identity.clone();
    dev.apply_cnot(alphabet.gates()[0]).unwrap();
    let test = MatrixState::from_packed_rows(DIMENSION, alphabet.id(), vec![2, 4, 1, 8]).unwrap();
    let forbidden = finite_views(&dev, &paths)
        .into_iter()
        .chain(finite_views(&test, &paths))
        .collect::<HashSet<_>>();

    let mut states = paths.keys().cloned().collect::<Vec<_>>();
    states.sort_by(|left, right| left.packed_rows().cmp(right.packed_rows()));
    let train = states
        .iter()
        .find(|candidate| {
            !forbidden.contains(*candidate)
                && has_supported_partial_shear(candidate)
                && alphabet.gates().iter().all(|gate| {
                    let mut neighbor = (*candidate).clone();
                    neighbor.apply_cnot(*gate).unwrap();
                    !forbidden.contains(&neighbor) && has_supported_partial_shear(&neighbor)
                })
        })
        .cloned()
        .expect("train root and its one-CNOT successors must stay outside holdouts and have shear");

    let directory = std::env::temp_dir().join(format!(
        "cnot103-cli-run-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&directory).unwrap();

    let mut catalog_entries = Vec::with_capacity(paths.len());
    for state in &states {
        let path = paths.get(state).unwrap();
        let reduction = path.iter().rev().collect::<Vec<_>>();
        catalog_entries.push(json!({
            "packed_rows": state.packed_rows(), "distance": path.len(),
            "gates": reduction.iter().map(|gate| json!({"control": gate.control, "target": gate.target})).collect::<Vec<_>>(),
            "producer": "integration-test-bfs-gl4", "lineage_id": format!("catalog-{:?}", state.packed_rows())
        }));
    }
    let catalog_digest = write_pinned_json(
        &directory,
        "catalogs.json",
        &json!({
            "schema": "exact_distance_catalog_bundle_v1",
            "provenance": {"fixture": "exhaustive GL(4,2) BFS"},
            "catalogs": [{"dimension": DIMENSION, "entries": catalog_entries}]
        }),
    );

    let mut forbidden_rows = forbidden.into_iter().collect::<Vec<_>>();
    forbidden_rows.sort_by(|left, right| left.packed_rows().cmp(right.packed_rows()));
    let dataset_digest = write_pinned_json(
        &directory,
        "dataset.json",
        &json!({
            "schema": "own_frozen_panels_v1",
            "provenance": {"fixture": "full CUDA CLI resume integration"},
            "symmetries": "identity,inverse,transpose,inverse_transpose and wire reversal; not full orbit",
            "forbidden_keys": forbidden_rows.iter().map(|state| json!([
                state.dimension(), state.alphabet_id(), state.packed_rows()
            ])).collect::<Vec<_>>(),
            "historical_keys": [], "rejected_train_lineages": [],
            "panels": {
                "train": [record_json(&train, &paths, "train")],
                "dev": [record_json(&dev, &paths, "dev")],
                "test": [record_json(&test, &paths, "test")]
            }
        }),
    );

    let gate_values = alphabet
        .gates()
        .iter()
        .map(|gate| json!([gate.control, gate.target]))
        .collect::<Vec<_>>();
    let gate_specs = alphabet
        .gates()
        .iter()
        .map(|gate| json!({"control": gate.control, "target": gate.target}))
        .collect::<Vec<_>>();
    let contexts = json!({
        "max_n": DIMENSION,
        "labeled_cuts": [{"n":DIMENSION,"a":[0,1]}],
        "restricted_intersections": [{"n":DIMENSION,"left":[0],"right":[1],"coordinates":[0]}],
        "rank_increment_probes": [{"n":DIMENSION,"span_rows":[0],"coordinates":[0],"candidate_support":[1]}],
        "alphabet": {"dimension":DIMENSION,"id":alphabet.id(),"gates":gate_specs,"costs":vec![1; alphabet.gates().len()]},
        "selected_residual_gates":[{"control":0,"target":1}],
        "block_parity_contexts":[{"n":DIMENSION,"sources":[0,1],"targets":[2,3]}],
        "max_parity_patterns":1024,"max_parity_pairs":4096,
        "matrix_boundaries":[{
            "n":DIMENSION,"inputs":[0,1],"outputs":[2,3],"protected_inputs":[],"zero_outputs":[],
            "weights":[
                [{"numerator":"1","denominator":"1"},{"numerator":"1","denominator":"1"}],
                [{"numerator":"1","denominator":"1"},{"numerator":"1","denominator":"1"}]
            ]
        }],
        "max_boundary_ports":8,"max_boundary_equations":1024,
        "paid_basis_contexts":[{"dimension":DIMENSION,"source_wires":[0,1],"target_wires":[2,3]}],
        "paid_shear_config":{"local_width":2,"sweeps":1},
        "template_response":{
            "choices":[{"kind":"add","control":0,"target":1},{"kind":"add","control":1,"target":0}],
            "points":[[{"numerator":1,"denominator":2},{"numerator":1,"denominator":2}]],
            "budgets":{"max_width":3,"max_assignments":16,"max_evaluations":128,"max_bytes":65536}
        },
        "bilinear_contexts":[{
            "n":DIMENSION,"rows":[0],"columns":[2],"hidden_width":1,
            "u_rows":[[1]],"v_rows":[[1]],"q":1,"a0_rows":[[0]],"d0_rows":[[0]]
        }],
        "shared_relation_contexts":[{"n":DIMENSION,"rows":[0],"coordinates":[1],"covectors":[[1]]}],
        "frequency_contexts":[{
            "n":DIMENSION,"sources":[0],"internal":[1,2],"targets":[3],"central_rows":[[1]]
        }],
        "positive_packet_contexts":[{"n":DIMENSION,"gates":[[0,2]],"factors":[{"scope":[0],"allowed":["0","1"]}]}],
        "i09_contexts":[{"n":DIMENSION,"horizon":1,"cut":0,"prefix_bits":[],"allowed_gates":gate_values}],
        "i09_limits":{"max_variables":1024,"max_equations":4096,"max_bytes":1048576},
        "i10_config":{"max_n":DIMENSION,"max_degree":12,"max_layer_states":25000,"max_transitions":3000000}
    });
    let mut manifest = json!({
        "schema":"cnot103_training_run_v1",
        "dataset":{"path":"dataset.json","sha256":dataset_digest},
        "exact_catalogs":{"path":"catalogs.json","sha256":catalog_digest},
        "checkpoint_dir":"checkpoint","resume":false,
        "training":{"seed":1234,"cuda_device":0,"attempts":1,"batch_size":1,"checkpoint_every_attempts":1},
        "feature_policy":{"schema_id":"runtime_packet_layout_v24","solver_assisted":true,"evaluations":null,"max_bytes":null},
        "model":{"wire_dim":8,"pair_dim":4,"structural_blocks":2,"attention_heads":2,"feedforward_dim":16,"fusion_blocks":1,"basis_proposal_hidden_dim":8},
        "optimizer":{"learning_rate":0.0001,"beta1":0.9,"beta2":0.999,"epsilon":0.00000001,"weight_decay":0.01},
        "objective":{"bellman_weight":1.0,"supervision_weight":1.0,"qhat_weight":0.1,"teacher_weight":0.0,"neighbor_chunk_size":8,"teacher_temperature":4.0},
        "alphabets":[contexts["alphabet"].clone()],"feature_contexts":contexts
    });
    let manifest_path = directory.join("run.json");

    manifest["resume"] = json!(false);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let first = run_cli(&manifest_path);
    assert_eq!(first["start_progress"]["attempted"], 0);
    assert_eq!(first["end_progress"]["attempted"], 1);
    assert_eq!(first["feature_model_inputs"], 102);
    assert_eq!(first["exposure_audit"]["strict_unseen"], true);

    manifest["resume"] = json!(true);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let resumed = run_cli(&manifest_path);
    assert_eq!(resumed["start_progress"]["attempted"], 1);
    assert_eq!(resumed["end_progress"]["attempted"], 2);
    assert_eq!(resumed["dataset_sha256"], first["dataset_sha256"]);
    assert_eq!(resumed["objective_sha256"], first["objective_sha256"]);
    fs::remove_dir_all(directory).unwrap();
}
