//! End-to-end reconstruction and checkpoint-resume tests for training sessions.

use super::*;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::ModelConfig;
use crate::nn_training::{AdamWConfig, NeuralTrainer};
use crate::objective::FullGraphObjectiveConfig;
use crate::schema::FeatureId;
use crate::training::Sampler;
use crate::{CnotAlphabet, CnotGate, MatrixState};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

static TORCH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
use tch::{nn::VarStore, Device};

const SEED: i64 = 2043;
const CONFIG: TrainingSessionConfig = TrainingSessionConfig {
    batch_size: 1,
    checkpoint_every_attempts: 1,
};

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cnot103-training-session-{label}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn state_json(state: &MatrixState) -> Value {
    let packed_rows = state
        .packed_rows()
        .chunks(state.words_per_row())
        .map(|row| {
            let value = row.iter().enumerate().fold(0u128, |acc, (word, bits)| {
                acc | (u128::from(*bits) << (word * 64))
            });
            json!(value)
        })
        .collect::<Vec<_>>();
    json!({
        "n": state.dimension(),
        "packed_rows": packed_rows,
        "alphabet_id": state.alphabet_id(),
    })
}

fn key_json(state: &MatrixState) -> Value {
    let packed_rows = state
        .packed_rows()
        .chunks(state.words_per_row())
        .map(|row| {
            row.iter().enumerate().fold(0u128, |acc, (word, bits)| {
                acc | (u128::from(*bits) << (word * 64))
            })
        })
        .collect::<Vec<_>>();
    json!([state.dimension(), state.alphabet_id(), packed_rows])
}

fn one_gate_record(n: u32, split: &str) -> (Value, Vec<MatrixState>) {
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let identity = MatrixState::identity(n, alphabet.id()).unwrap();
    let gate = alphabet.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let lineage = format!("{split}-lineage");
    let record = json!({
        "state": state_json(&state),
        "verified_words": [{
            "start": state_json(&state),
            "end": state_json(&identity),
            "gates": [{"control": gate.control, "target": gate.target}],
            "physical_cost": 1,
            "producer": "training-session-fixture",
            "lineage_id": lineage,
        }],
        "lineage_id": lineage,
        "seed": n,
        "source": format!("source-{split}"),
        "split": split,
        "exact_distance": null,
        "exact_evidence": null,
    });
    let mut views = HashSet::new();
    let reversed_control = n - 1 - gate.control;
    let reversed_target = n - 1 - gate.target;
    for transformed in [
        gate,
        CnotGate {
            control: gate.target,
            target: gate.control,
        },
        CnotGate {
            control: reversed_control,
            target: reversed_target,
        },
        CnotGate {
            control: n - 1 - gate.target,
            target: n - 1 - gate.control,
        },
    ] {
        let mut view = identity.clone();
        view.apply_cnot(transformed).unwrap();
        views.insert(view);
    }
    (record, views.into_iter().collect())
}

fn frozen_fixture(path: &Path) -> String {
    let (train, _) = one_gate_record(4, "train");
    let (dev, dev_views) = one_gate_record(3, "dev");
    let (test, test_views) = one_gate_record(2, "test");
    let forbidden = dev_views
        .into_iter()
        .chain(test_views)
        .collect::<HashSet<_>>();
    let body = json!({
        "schema": "own_frozen_panels_v1",
        "provenance": {"fixture": "training_session"},
        "symmetries": "identity,inverse,transpose,inverse_transpose and wire reversal; not full orbit",
        "forbidden_keys": forbidden.iter().map(key_json).collect::<Vec<_>>(),
        "historical_keys": [],
        "rejected_train_lineages": [],
        "panels": {"train": [train], "dev": [dev], "test": [test]},
    });
    let bytes = serde_json::to_vec(&body).unwrap();
    std::fs::write(path, &bytes).unwrap();
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    std::fs::write(path.with_extension("sha256"), format!("{digest}\n")).unwrap();
    digest
}

struct Components {
    dataset: FrozenDataset,
    variables: VarStore,
    model: FusionDistanceField,
    registry: FeatureRegistry,
    objective: FullGraphObjective,
    trainer: NeuralTrainer,
}

fn first_parameter_difference(
    expected: &std::collections::HashMap<String, tch::Tensor>,
    actual: &VarStore,
) -> Option<String> {
    let actual = actual.variables();
    if expected.len() != actual.len() {
        return Some(format!(
            "parameter count {} != {}",
            expected.len(),
            actual.len()
        ));
    }
    let mut names = expected.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let Some(other) = actual.get(&name) else {
            return Some(format!("missing parameter {name}"));
        };
        if !expected[&name].equal(other) {
            return Some(format!("parameter values differ: {name}"));
        }
    }
    None
}

fn first_model_payload_difference(left: &[u8], right: &[u8]) -> Option<String> {
    let left = tch::Tensor::load_multi_from_stream_with_device(Cursor::new(left), Device::Cpu)
        .unwrap()
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    let right = tch::Tensor::load_multi_from_stream_with_device(Cursor::new(right), Device::Cpu)
        .unwrap()
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    if left.len() != right.len() {
        return Some(format!("tensor count {} != {}", left.len(), right.len()));
    }
    let mut names = left.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let Some(other) = right.get(&name) else {
            return Some(format!("missing tensor {name}"));
        };
        if !left[&name].equal(other) {
            return Some(format!("tensor values differ: {name}"));
        }
    }
    None
}

fn first_optimizer_state_difference(left: &[u8], right: &[u8]) -> Option<String> {
    const MAGIC: &[u8; 8] = b"C103ADW1";
    if left.len() < 16 || right.len() < 16 || &left[..8] != MAGIC || &right[..8] != MAGIC {
        return Some("optimizer state header is malformed".into());
    }
    let left_header_len = u64::from_le_bytes(left[8..16].try_into().unwrap()) as usize;
    let right_header_len = u64::from_le_bytes(right[8..16].try_into().unwrap()) as usize;
    let left_payload = 16usize.checked_add(left_header_len)?;
    let right_payload = 16usize.checked_add(right_header_len)?;
    if left_payload > left.len() || right_payload > right.len() {
        return Some("optimizer state header is truncated".into());
    }
    let left_header: Value = serde_json::from_slice(&left[16..left_payload]).ok()?;
    let right_header: Value = serde_json::from_slice(&right[16..right_payload]).ok()?;
    if left_header != right_header {
        return Some("optimizer metadata differs".into());
    }
    let left_tensors = tch::Tensor::load_multi_from_stream_with_device(
        Cursor::new(&left[left_payload..]),
        Device::Cpu,
    )
    .ok()?
    .into_iter()
    .collect::<std::collections::HashMap<_, _>>();
    let right_tensors = tch::Tensor::load_multi_from_stream_with_device(
        Cursor::new(&right[right_payload..]),
        Device::Cpu,
    )
    .ok()?
    .into_iter()
    .collect::<std::collections::HashMap<_, _>>();
    if left_tensors.len() != right_tensors.len() {
        return Some(format!(
            "optimizer tensor count {} != {}",
            left_tensors.len(),
            right_tensors.len()
        ));
    }
    let mut names = left_tensors.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let Some(other) = right_tensors.get(&name) else {
            return Some(format!("optimizer tensor missing: {name}"));
        };
        if !left_tensors[&name].equal(other) {
            return Some(format!("optimizer tensor values differ: {name}"));
        }
    }
    None
}

fn model_values(variables: &VarStore) -> std::collections::HashMap<String, tch::Tensor> {
    variables
        .variables()
        .into_iter()
        .map(|(name, value)| (name, value.detach().copy()))
        .collect()
}

fn model_state_bytes(variables: &VarStore) -> Vec<u8> {
    let mut bytes = Vec::new();
    variables.save_to_stream(&mut bytes).unwrap();
    bytes
}

fn components(dataset_path: &Path) -> Components {
    tch::set_num_threads(1);
    tch::manual_seed(SEED);
    let dataset = FrozenDataset::load(dataset_path).unwrap();
    let registry = FeatureRegistry::new();
    let mut policy = FeaturePolicy::complete_103("training-session-fixture-v1");
    let candidate_ids = policy.enabled_ids.clone();
    policy.enabled_ids = candidate_ids
        .into_iter()
        .filter(|id| {
            let mut one_id = policy.clone();
            one_id.enabled_ids = vec![*id];
            registry.selected_specs(&one_id).is_ok()
        })
        .collect();
    assert!(!policy.enabled_ids.is_empty());
    assert!(policy
        .enabled_ids
        .iter()
        .all(|id| *id != FeatureId::new(b'H', 7).unwrap()));
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let variables = VarStore::new(Device::Cpu);
    let model = FusionDistanceField::new_for_policy(
        &variables.root(),
        &registry,
        &policy,
        Device::Cpu,
        ModelConfig {
            wire_dim: 8,
            pair_dim: 4,
            structural_blocks: 1,
            attention_heads: 2,
            feedforward_dim: 16,
            fusion_blocks: 1,
            basis_proposal_hidden_dim: 8,
        },
    )
    .unwrap();
    let objective = FullGraphObjective::new(
        vec![
            CnotAlphabet::all_to_all_unit(2).unwrap(),
            CnotAlphabet::all_to_all_unit(3).unwrap(),
            alphabet,
        ],
        &registry,
        policy,
        Vec::new(),
        dataset.forbidden_keys.iter().cloned().collect(),
        FullGraphObjectiveConfig {
            qhat_weight: 0.0,
            teacher_weight: 0.0,
            require_all_model_inputs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let trainer = NeuralTrainer::new(
        &variables,
        Sampler::new(dataset.train.len(), 991).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    Components {
        dataset,
        variables,
        model,
        registry,
        objective,
        trainer,
    }
}

#[test]
fn reconstructed_session_resume_matches_continuous_two_attempt_run() {
    let _guard = TORCH_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    const CHILD_DATASET: &str = "CNOT103_SESSION_CHILD_DATASET";
    const CHILD_STORE: &str = "CNOT103_SESSION_CHILD_STORE";

    if let Some(dataset_path) = std::env::var_os(CHILD_DATASET) {
        let dataset_path = PathBuf::from(dataset_path);
        let store = CheckpointStore::new(PathBuf::from(std::env::var_os(CHILD_STORE).unwrap()));
        let mut second_process = components(&dataset_path);
        let report = {
            let Components {
                dataset,
                objective,
                model,
                variables,
                registry,
                trainer,
            } = &mut second_process;
            let mut session = FullGraphTrainingSession::resume_latest(
                TrainingSessionResources {
                    dataset,
                    objective,
                    model,
                    variables,
                    registry,
                    trainer,
                    store: &store,
                    admission: None,
                },
                CONFIG,
            )
            .unwrap();
            assert_eq!(session.progress().attempted, 1);
            session.run_attempts(1).unwrap()
        };
        assert_eq!(report.end_progress.attempted, 2);
        return;
    }

    let data_dir = TestDir::new("data");
    let dataset_path = data_dir.path().join("frozen.json");
    let pin = frozen_fixture(&dataset_path);

    let continuous_dir = TestDir::new("continuous");
    let continuous_store = CheckpointStore::new(continuous_dir.path());
    let mut continuous = components(&dataset_path);
    let continuous_initial_bytes = model_state_bytes(&continuous.variables);
    let continuous_initial_values = model_values(&continuous.variables);
    let (continuous_first_sha, continuous_report) = {
        let Components {
            dataset,
            objective,
            model,
            variables,
            registry,
            trainer,
        } = &mut continuous;
        let mut session = FullGraphTrainingSession::new(
            TrainingSessionResources {
                dataset,
                objective,
                model,
                variables,
                registry,
                trainer,
                store: &continuous_store,
                admission: None,
            },
            CONFIG,
        )
        .unwrap();
        let first = session.run_attempts(1).unwrap();
        let second = session.run_attempts(1).unwrap();
        assert_eq!(second.end_progress.attempted, 2);
        (first.final_checkpoint, second)
    };

    let continuous_first_checkpoint = continuous_store
        .load_manifest(
            &continuous_first_sha,
            &continuous.objective.checkpoint_contract(),
        )
        .unwrap();

    let resumed_dir = TestDir::new("resumed");
    let resumed_store = CheckpointStore::new(resumed_dir.path());
    let mut first_process = components(&dataset_path);
    first_process
        .variables
        .load_from_stream(Cursor::new(continuous_initial_bytes))
        .unwrap();
    assert_eq!(
        first_parameter_difference(&continuous_initial_values, &first_process.variables),
        None,
        "copied initial model differs"
    );
    let split_first_sha = {
        let Components {
            dataset,
            objective,
            model,
            variables,
            registry,
            trainer,
        } = &mut first_process;
        let mut session = FullGraphTrainingSession::new(
            TrainingSessionResources {
                dataset,
                objective,
                model,
                variables,
                registry,
                trainer,
                store: &resumed_store,
                admission: None,
            },
            CONFIG,
        )
        .unwrap();
        let first = session.run_attempts(1).unwrap();
        assert_eq!(first.end_progress.attempted, 1);
        first.final_checkpoint
    };
    let split_first_checkpoint = resumed_store
        .load_manifest(
            &split_first_sha,
            &first_process.objective.checkpoint_contract(),
        )
        .unwrap();
    drop(first_process);
    assert_eq!(
        continuous_first_checkpoint.sampler, split_first_checkpoint.sampler,
        "sampler differs"
    );
    assert_eq!(
        continuous_first_checkpoint.rng_state, split_first_checkpoint.rng_state,
        "rng differs"
    );
    assert_eq!(
        continuous_first_checkpoint.objective_state, split_first_checkpoint.objective_state,
        "objective ledger differs"
    );
    assert_eq!(
        continuous_first_checkpoint.optimizer_state, split_first_checkpoint.optimizer_state,
        "optimizer differs"
    );
    assert_eq!(
        first_model_payload_difference(
            &continuous_first_checkpoint.model_state,
            &split_first_checkpoint.model_state,
        ),
        None,
        "first-step model tensors differ"
    );

    let child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("training_session::tests::reconstructed_session_resume_matches_continuous_two_attempt_run")
            .arg("--nocapture")
            .env(CHILD_DATASET, &dataset_path)
            .env(CHILD_STORE, resumed_store.root())
            .output()
            .unwrap();
    assert!(
        child.status.success(),
        "fresh-process resume failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );

    let mut second_process = components(&dataset_path);
    second_process
        .objective
        .bind_frozen_dataset(&second_process.dataset)
        .unwrap();
    let resumed_manifest: CheckpointManifest =
        serde_json::from_slice(&std::fs::read(resumed_store.root().join("last.json")).unwrap())
            .unwrap();
    let resumed_final = resumed_store
        .load_manifest(
            &resumed_manifest,
            &second_process.objective.checkpoint_contract(),
        )
        .unwrap();

    assert_eq!(pin, continuous.dataset.sha256);
    assert_eq!(
        continuous_first_sha.contract_sha256, split_first_sha.contract_sha256,
        "first checkpoint contracts differ"
    );
    assert_eq!(continuous_first_sha.progress, split_first_sha.progress);
    assert_eq!(
        continuous_first_sha.sha256, split_first_sha.sha256,
        "equivalent first-step checkpoints must have canonical bytes"
    );
    let continuous_final = continuous_store
        .load_manifest(
            &continuous_report.final_checkpoint,
            &continuous.objective.checkpoint_contract(),
        )
        .unwrap();
    assert_eq!(
        first_model_payload_difference(&continuous_final.model_state, &resumed_final.model_state),
        None,
        "final model tensors differ after resume"
    );
    assert_eq!(continuous_final.sampler, resumed_final.sampler);
    assert_eq!(
        first_optimizer_state_difference(
            &continuous_final.optimizer_state,
            &resumed_final.optimizer_state,
        ),
        None,
        "final optimizer state differs after fresh-process resume"
    );
    assert!(
        continuous_final.objective_state == resumed_final.objective_state,
        "final objective state differs"
    );
    assert_eq!(continuous_final.rng_state, resumed_final.rng_state);
    assert_eq!(continuous_report.end_progress, resumed_final.progress);
    assert_eq!(
        continuous_report.final_checkpoint.sha256, resumed_manifest.sha256,
        "resumed two-step checkpoint must match uninterrupted bytes"
    );
}
