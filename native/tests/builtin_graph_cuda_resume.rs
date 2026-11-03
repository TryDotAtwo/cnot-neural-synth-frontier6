use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::nn_training::{AdamWConfig, NeuralTrainer};
use cnot103_native::objective::{
    FullGraphObjective, FullGraphObjectiveConfig, FullGraphObjectiveRunner,
};
use cnot103_native::training_data::{TrainingRecord, TrainingSplit};
use cnot103_native::{
    evaluate_h07_prediction_spread, CheckpointStore, CnotAlphabet, CnotGate, FeatureId,
    FeaturePolicy, FeatureRegistry, H07ModelProbe, H07ProbeKind, MatrixState, Sampler,
    WordCertificate,
};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tch::nn::VarStore;
use tch::{Device, Kind};

#[path = "support/feature_contexts.rs"]
mod feature_contexts;
use feature_contexts::complete_model_input_contexts;

#[test]
fn complete_model_input_profile_is_accepted_by_default_training_gate() {
    let registry = FeatureRegistry::with_all_model_input_providers(complete_model_input_contexts())
        .expect("register all 102 model-input providers");
    let mut policy =
        FeaturePolicy::all_model_inputs_augmentation_only("complete-model-inputs-contract-test");
    policy.solver_assisted = true;
    let selected = registry.selected_specs(&policy).unwrap();
    assert_eq!(selected.len(), cnot103_native::schema::MODEL_INPUT_COUNT);
    assert!(!policy
        .enabled_ids
        .contains(&FeatureId::new(b'H', 7).unwrap()));

    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    FullGraphObjective::new(
        vec![alphabet],
        &registry,
        policy,
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .expect("default complete-model-input gate must accept H07 as diagnostic-only");
}

const DEVICE_ENV: &str = "CNOT103_RESUME_DEVICE";
const STORE_ENV: &str = "CNOT103_BUILTIN_RESUME_STORE";
const OUTPUT_ENV: &str = "CNOT103_BUILTIN_RESUME_OUTPUT";
const PROFILE_ENV: &str = "CNOT103_CUDA_RESUME_PROFILE";
static CUDA_RESUME_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
const MODEL_SEED: i64 = 2043;
const SAMPLER_SEED: u64 = 2043;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeatureProfile {
    Builtin41,
    AllModelInputs102,
}

impl FeatureProfile {
    fn as_env(self) -> &'static str {
        match self {
            Self::Builtin41 => "builtin-41",
            Self::AllModelInputs102 => "all-model-inputs-102",
        }
    }

    fn from_env() -> Self {
        match std::env::var(PROFILE_ENV).as_deref() {
            Ok("all-model-inputs-102") => Self::AllModelInputs102,
            Ok("builtin-41") | Err(_) => Self::Builtin41,
            Ok(other) => panic!("unsupported {PROFILE_ENV} value: {other}"),
        }
    }

    fn feature_count(self) -> usize {
        match self {
            Self::Builtin41 => 41,
            Self::AllModelInputs102 => 102,
        }
    }
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before UNIX epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cnot103-builtin-graph-resume-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create isolated checkpoint directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    variables: VarStore,
    model: FusionDistanceField,
    registry: FeatureRegistry,
    policy: FeaturePolicy,
    alphabet: CnotAlphabet,
    records: Vec<TrainingRecord>,
    objective: FullGraphObjective,
    trainer: NeuralTrainer,
}

fn selected_cuda_index() -> Option<usize> {
    let index = match std::env::var(DEVICE_ENV).as_deref() {
        Ok("cuda:0") | Err(_) => 0,
        Ok("cuda:1") => 1,
        Ok(other) => panic!("unsupported {DEVICE_ENV} value: {other}"),
    };
    match cnot103_native::torch_device::require_cuda_device(index) {
        Ok(()) => Some(index),
        Err(error) if std::env::var_os("CNOT103_REQUIRE_CUDA").is_some() => {
            panic!("CNOT103_REQUIRE_CUDA is set but cuda:{index} is unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable; skipping full-graph resume test: {error}");
            None
        }
    }
}

fn fixture(device: Device, profile: FeatureProfile) -> Fixture {
    tch::set_num_threads(1);
    tch::manual_seed(MODEL_SEED);

    let (registry, policy) = match profile {
        FeatureProfile::Builtin41 => {
            let registry = FeatureRegistry::new();
            let mut policy = FeaturePolicy::complete_103("builtin-41-full-graph-resume-v1");
            let candidate_ids = policy.enabled_ids.clone();
            policy.enabled_ids = candidate_ids
                .into_iter()
                .filter(|id| {
                    let mut one_id = policy.clone();
                    one_id.enabled_ids = vec![*id];
                    registry.selected_specs(&one_id).is_ok()
                })
                .collect();
            (registry, policy)
        }
        FeatureProfile::AllModelInputs102 => {
            let registry =
                FeatureRegistry::with_all_model_input_providers(complete_model_input_contexts())
                    .expect("register all 102 model-input providers from explicit contexts");
            let mut policy =
                FeaturePolicy::all_model_inputs_augmentation_only("all-model-inputs-102-resume-v1");
            policy.solver_assisted = true;
            (registry, policy)
        }
    };
    let specs = registry
        .selected_specs(&policy)
        .expect("the selected feature profile is executable");
    assert_eq!(
        specs.len(),
        profile.feature_count(),
        "feature profile must register its full declared input set"
    );

    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let identity = MatrixState::identity(3, alphabet.id()).unwrap();
    let gate = alphabet.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let records = vec![TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![WordCertificate {
            start: state.clone(),
            end: identity,
            gates: vec![gate],
            physical_cost: 1,
            producer: "cuda_resume_one_gate_fixture".into(),
            lineage_id: "cuda-resume-fixture-word".into(),
        }],
        exact_distance: None,
        lineage_id: "cuda-resume-fixture-record".into(),
    }];

    let variables = VarStore::new(device);
    let model = FusionDistanceField::new(
        &variables.root(),
        &specs,
        policy.schema_id.clone(),
        policy.id().unwrap(),
        device,
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
        vec![alphabet.clone()],
        &registry,
        policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig {
            qhat_weight: 0.0,
            teacher_weight: 0.0,
            require_all_model_inputs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut trainer = NeuralTrainer::new(
        &variables,
        Sampler::new(records.len(), SAMPLER_SEED).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    trainer.set_rng_state(b"cuda-builtin-full-graph-resume-v1".to_vec());

    Fixture {
        variables,
        model,
        registry,
        policy,
        alphabet,
        records,
        objective,
        trainer,
    }
}

fn run_updates(fixture: &mut Fixture, count: usize) {
    let mut runner = FullGraphObjectiveRunner::new(
        &mut fixture.objective,
        &fixture.model,
        &fixture.variables,
        &fixture.registry,
        &fixture.records,
        None,
    )
    .expect("bind the full-graph objective to the 41-feature registry and records");
    for _ in 0..count {
        let outcome = fixture
            .trainer
            .step_with_objective(1, &mut runner)
            .expect("run native Bellman training update");
        assert!(outcome.accepted, "Bellman update must be accepted");
        assert!(outcome.loss.is_finite(), "Bellman loss must remain finite");
    }
}

fn summary(fixture: &Fixture) -> Value {
    let mut variables: Vec<_> = fixture.variables.variables().into_iter().collect();
    variables.sort_by(|left, right| left.0.cmp(&right.0));
    let parameters: Vec<_> = variables
        .into_iter()
        .map(|(name, tensor)| {
            let shape = tensor.size();
            let flat = tensor
                .detach()
                .to_device(Device::Cpu)
                .to_kind(Kind::Double)
                .contiguous()
                .view([-1]);
            let mut values = vec![0.0f64; flat.numel()];
            let count = values.len();
            flat.copy_data(&mut values, count);
            json!({"name": name, "shape": shape, "values": values})
        })
        .collect();

    json!({
        "parameters": parameters,
        "progress": fixture.trainer.progress(),
        "optimizer_steps": fixture.trainer.optimizer_steps(),
        "sampler": fixture.trainer.sampler_snapshot(),
        "objective": fixture.objective.state_dict(),
        "rng_state": fixture.trainer.rng_state(),
        "feature_count": fixture.policy.enabled_ids.len(),
        "alphabet": fixture.alphabet.id(),
    })
}

fn assert_equivalent(actual: &Value, expected: &Value) {
    for field in [
        "progress",
        "optimizer_steps",
        "sampler",
        "objective",
        "rng_state",
        "feature_count",
        "alphabet",
    ] {
        assert_eq!(
            actual[field], expected[field],
            "resume field differs: {field}"
        );
    }
    let actual_parameters = actual["parameters"].as_array().unwrap();
    let expected_parameters = expected["parameters"].as_array().unwrap();
    assert_eq!(actual_parameters.len(), expected_parameters.len());
    let mut max_abs_error = 0.0f64;
    for (actual, expected) in actual_parameters.iter().zip(expected_parameters) {
        assert_eq!(actual["name"], expected["name"]);
        assert_eq!(actual["shape"], expected["shape"]);
        let actual_values = actual["values"].as_array().unwrap();
        let expected_values = expected["values"].as_array().unwrap();
        assert_eq!(actual_values.len(), expected_values.len());
        for (actual, expected) in actual_values.iter().zip(expected_values) {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            max_abs_error = max_abs_error.max((actual - expected).abs());
        }
    }
    assert!(
        max_abs_error <= 1e-6,
        "fresh-process parameters differ from uninterrupted training: max_abs_error={max_abs_error}"
    );
}

#[test]
fn fresh_process_full_graph_cuda_resume_matches_uninterrupted_builtin_features() {
    let _guard = CUDA_RESUME_TEST_LOCK
        .lock()
        .expect("serialize CUDA resume profiles around process-global LibTorch RNG");
    run_resume_case(FeatureProfile::Builtin41);
}

#[test]
fn fresh_process_full_graph_cuda_resume_matches_uninterrupted_all_model_inputs() {
    let _guard = CUDA_RESUME_TEST_LOCK
        .lock()
        .expect("serialize CUDA resume profiles around process-global LibTorch RNG");
    run_resume_case(FeatureProfile::AllModelInputs102);
}

fn run_resume_case(profile: FeatureProfile) {
    let Some(device_index) = selected_cuda_index() else {
        return;
    };
    let device = Device::Cuda(device_index);
    let temp = TestDir::new();
    let store = CheckpointStore::new(temp.path().join("checkpoint"));

    let mut uninterrupted = fixture(device, profile);
    run_updates(&mut uninterrupted, 2);
    let expected = summary(&uninterrupted);

    let mut split = fixture(device, profile);
    run_updates(&mut split, 1);
    {
        let runner = FullGraphObjectiveRunner::new(
            &mut split.objective,
            &split.model,
            &split.variables,
            &split.registry,
            &split.records,
            None,
        )
        .unwrap();
        let manifest = runner
            .save_checkpoint(&mut split.trainer, &store, false)
            .expect("save the model, AdamW, sampler, and objective checkpoint");
        assert_eq!(manifest.progress.accepted, 1);
    }

    let output = temp.path().join("fresh-process-summary.json");
    let child = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--ignored")
        .arg("--exact")
        .arg("fresh_process_full_graph_cuda_resume_child")
        .arg("--nocapture")
        .env(DEVICE_ENV, format!("cuda:{device_index}"))
        .env(PROFILE_ENV, profile.as_env())
        .env("CNOT103_REQUIRE_CUDA", "1")
        .env(STORE_ENV, store.root())
        .env(OUTPUT_ENV, &output)
        .output()
        .expect("launch a fresh process for the restored update");
    assert!(
        child.status.success(),
        "fresh process failed
stdout:
{}
stderr:
{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    let actual: Value = serde_json::from_slice(
        &fs::read(output).expect("read the fresh-process model and training summary"),
    )
    .expect("parse fresh-process summary");
    assert_equivalent(&actual, &expected);
    assert_eq!(actual["progress"]["accepted"], 2);
    assert_eq!(actual["optimizer_steps"], 2);
    assert_eq!(actual["feature_count"], profile.feature_count());
}

fn state_from_word(alphabet: &CnotAlphabet, gates: &[CnotGate]) -> MatrixState {
    let mut state = MatrixState::identity(alphabet.dimension(), alphabet.id()).unwrap();
    for gate in gates {
        state.apply_cnot(*gate).unwrap();
    }
    state
}

fn relabel_state(state: &MatrixState, output_to_source: &[usize]) -> MatrixState {
    let n = state.dimension() as usize;
    let words_per_row = state.words_per_row();
    assert_eq!(output_to_source.len(), n);
    let mut rows = vec![0u64; n * words_per_row];
    for output_row in 0..n {
        for output_column in 0..n {
            let source_row = output_to_source[output_row];
            let source_column = output_to_source[output_column];
            let source_words = state.row_words(source_row as u32).unwrap();
            let bit = (source_words[source_column / 64] >> (source_column % 64)) & 1;
            if bit != 0 {
                rows[output_row * words_per_row + output_column / 64] |=
                    1u64 << (output_column % 64);
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), rows).unwrap()
}

#[test]
fn h07_runs_cuda_forwards_for_equivalent_view_and_relabel_packets() {
    let _guard = CUDA_RESUME_TEST_LOCK
        .lock()
        .expect("serialize CUDA resume and H07 probes around process-global LibTorch RNG");
    let Some(device_index) = selected_cuda_index() else {
        return;
    };
    let fixture = fixture(
        Device::Cuda(device_index),
        FeatureProfile::AllModelInputs102,
    );
    assert_eq!(
        fixture.policy.enabled_ids.len(),
        102,
        "H07 actual forwards must exercise all 102 model inputs"
    );
    let gates = [
        CnotGate {
            control: 0,
            target: 1,
        },
        CnotGate {
            control: 1,
            target: 2,
        },
        CnotGate {
            control: 2,
            target: 0,
        },
    ];
    let canonical = state_from_word(&fixture.alphabet, &gates);
    let inverse_gates = gates.into_iter().rev().collect::<Vec<_>>();
    let inverse = state_from_word(&fixture.alphabet, &inverse_gates);
    let relabeled = relabel_state(&canonical, &[1, 0, 2]);
    let states = [canonical, inverse, relabeled];
    let packets = states
        .iter()
        .map(|state| {
            fixture
                .registry
                .build(std::slice::from_ref(state), &fixture.policy)
                .expect("build each exact-view feature packet")
        })
        .collect::<Vec<_>>();
    let probes = [
        H07ModelProbe {
            probe_id: "canonical".into(),
            kind: H07ProbeKind::Canonical,
            packet: &packets[0],
        },
        H07ModelProbe {
            probe_id: "inverse".into(),
            kind: H07ProbeKind::Inverse,
            packet: &packets[1],
        },
        H07ModelProbe {
            probe_id: "wire-swap-01".into(),
            kind: H07ProbeKind::Relabel,
            packet: &packets[2],
        },
    ];

    let diagnostic =
        evaluate_h07_prediction_spread(&fixture.model, &"1".repeat(64), &"2".repeat(64), &probes)
            .expect("run canonical, inverse-view, and cost-preserving relabel forwards");
    assert_eq!(diagnostic.sample_count, 3);
    assert_eq!(diagnostic.view_sample_count, 2);
    assert_eq!(diagnostic.relabel_sample_count, 1);
    assert!(diagnostic
        .probes
        .iter()
        .all(|probe| probe.prediction.is_finite()));
    for probe in &diagnostic.probes {
        let coverage = probe
            .input_coverage
            .as_ref()
            .expect("actual H07 forwards retain per-probe model-input coverage");
        assert_eq!(coverage.entries.len(), 102);
        assert_eq!(coverage.selected_model_input_count(), 102);
        assert!(
            coverage.unavailable_model_input_count() > 0,
            "the fixture's unsupported providers must remain visible in the H07 audit"
        );
        assert!(!coverage.is_complete());
    }
}

#[test]
#[ignore = "invoked in a new process by fresh_process_full_graph_cuda_resume_matches_uninterrupted_builtin_features"]
fn fresh_process_full_graph_cuda_resume_child() {
    let (Some(store), Some(output)) = (std::env::var_os(STORE_ENV), std::env::var_os(OUTPUT_ENV))
    else {
        return;
    };
    let device_index = selected_cuda_index().expect("parent requires a working CUDA device");
    let profile = FeatureProfile::from_env();
    let mut fixture = fixture(Device::Cuda(device_index), profile);
    let store = CheckpointStore::new(store);
    fixture
        .objective
        .restore_training_checkpoint(
            &mut fixture.trainer,
            &mut fixture.variables,
            &store,
            &fixture.records,
        )
        .expect("restore the fresh-process full-graph checkpoint");
    run_updates(&mut fixture, 1);
    fs::write(
        output,
        serde_json::to_vec(&summary(&fixture)).expect("encode restored training state"),
    )
    .expect("write fresh-process summary");
}
