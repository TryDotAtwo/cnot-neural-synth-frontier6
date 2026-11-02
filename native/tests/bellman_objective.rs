use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::nn_training::{AdamWConfig, NeuralTrainer};
use cnot103_native::objective::{
    FullGraphObjective, FullGraphObjectiveConfig, FullGraphObjectiveRunner, StateAdmission,
};
use cnot103_native::training_data::{TrainingRecord, TrainingSplit};
use cnot103_native::CheckpointStore;
use cnot103_native::{
    recompute_backup, select_backup, CnotAlphabet, CnotGate, FeatureComputation, FeatureId,
    FeaturePolicy, FeatureProducer, FeatureRegistry, FeatureRole, FeatureSpec, FeatureStatus,
    FeatureValue, MatrixState, Sampler, WordCertificate,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use tch::nn::VarStore;
use tch::{Device, Tensor};

struct ZeroFeature {
    spec: FeatureSpec,
}

struct AllowAdmission {
    fingerprint: String,
}

impl StateAdmission for AllowAdmission {
    fn contract_sha256(&self) -> &str {
        &self.fingerprint
    }

    fn admit(&self, _state: &MatrixState) -> Result<(), String> {
        Ok(())
    }
}

impl FeatureProducer for ZeroFeature {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        if self.spec.id == FeatureId::new(b'C', 2).unwrap() && state.is_identity() {
            return FeatureComputation::Unavailable {
                status: FeatureStatus::Inapplicable,
                reason: "fixture_only_for_nonidentity_states".into(),
            };
        }
        let n = state.dimension() as usize;
        let (shape, value_count) = match self.spec.role {
            FeatureRole::Node => (vec![n, self.spec.channels], n * self.spec.channels),
            FeatureRole::Pair => (vec![n, n, self.spec.channels], n * n * self.spec.channels),
            FeatureRole::Global => (vec![self.spec.channels], self.spec.channels),
            FeatureRole::Group => (vec![1, self.spec.channels], self.spec.channels),
        };
        FeatureComputation::Computed(FeatureValue {
            shape,
            values: vec![0.0; value_count],
            token_valid: if self.spec.role == FeatureRole::Group {
                vec![1]
            } else {
                Vec::new()
            },
            incidence: if self.spec.role == FeatureRole::Group {
                vec![0.0; n * self.spec.wire_roles.len()]
            } else {
                Vec::new()
            },
        })
    }
}

struct Fixture {
    variables: VarStore,
    model: FusionDistanceField,
    registry: FeatureRegistry,
    policy: FeaturePolicy,
    alphabet: CnotAlphabet,
}

fn fixture() -> Fixture {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut registry = FeatureRegistry::new();
    for mut spec in registry.specs() {
        if spec.id == FeatureId::new(b'H', 7).unwrap() || spec.producer_version != "unconfigured" {
            continue;
        }
        spec.scope = "bellman_fixture".into();
        spec.producer_version = format!("bellman_fixture_{}", spec.id);
        spec.dependencies = vec!["M".into()];
        spec.exactness = "exact".into();
        spec.cost_tier = "T1".into();
        if spec.role == FeatureRole::Group && spec.wire_roles.is_empty() {
            spec.wire_roles = vec!["fixture_wire".into()];
        }
        registry.register(Box::new(ZeroFeature { spec })).unwrap();
    }
    let mut policy = FeaturePolicy::all_model_inputs("bellman-fixture-v1");
    policy.solver_assisted = true;
    let policy_id = policy.id().unwrap();
    let model_specs = registry.selected_specs(&policy).unwrap();
    let variables = VarStore::new(Device::Cpu);
    let config = ModelConfig {
        wire_dim: 8,
        pair_dim: 4,
        structural_blocks: 1,
        attention_heads: 2,
        feedforward_dim: 16,
        fusion_blocks: 1,
        basis_proposal_hidden_dim: 8,
    };
    let model = FusionDistanceField::new(
        &variables.root(),
        &model_specs,
        policy.schema_id.clone(),
        policy_id,
        Device::Cpu,
        config,
    )
    .unwrap();
    Fixture {
        variables,
        model,
        registry,
        policy,
        alphabet,
    }
}

#[test]
fn full_neighbor_selection_is_chunk_invariant_and_recomputes_a_frozen_loss() {
    tch::set_num_threads(1);
    tch::manual_seed(2026);
    let fixture = fixture();
    let identity = MatrixState::identity(3, fixture.alphabet.id()).unwrap();
    let mut nonterminal = identity.clone();
    nonterminal
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let states = [identity.clone(), nonterminal.clone()];

    let parent_packet = fixture
        .registry
        .build(std::slice::from_ref(&identity), &fixture.policy)
        .unwrap();
    let inapplicable = parent_packet.feature(FeatureId::new(b'C', 2).unwrap());
    assert_eq!(inapplicable.available, vec![0]);
    assert_eq!(inapplicable.status, vec![FeatureStatus::Inapplicable]);
    assert!(fixture.model.forward(&parent_packet).is_ok());

    let mut expected_index = 0usize;
    let mut expected_total = f64::INFINITY;
    for (index, gate) in fixture.alphabet.gates().iter().copied().enumerate() {
        let mut successor = nonterminal.clone();
        successor.apply_cnot(gate).unwrap();
        let packet = fixture
            .registry
            .build(std::slice::from_ref(&successor), &fixture.policy)
            .unwrap();
        let value = tch::no_grad(|| {
            fixture
                .model
                .forward(&packet)
                .unwrap()
                .value
                .double_value(&[0])
        });
        let total = value + f64::from(fixture.alphabet.cost_at(index).unwrap());
        if total < expected_total {
            expected_index = index;
            expected_total = total;
        }
    }

    let chunked = select_backup(
        &fixture.model,
        &fixture.variables,
        &states,
        &fixture.registry,
        &fixture.policy,
        &fixture.alphabet,
        Some(1),
    )
    .unwrap();
    let unchunked = select_backup(
        &fixture.model,
        &fixture.variables,
        &states,
        &fixture.registry,
        &fixture.policy,
        &fixture.alphabet,
        None,
    )
    .unwrap();
    let chunked_audit: Vec<_> = chunked.branches().cloned().collect();
    let unchunked_audit: Vec<_> = unchunked.branches().cloned().collect();
    assert_eq!(chunked_audit[0].selected_index, None);
    assert_eq!(chunked_audit[1].selected_index, Some(expected_index));
    assert_eq!(chunked_audit.len(), unchunked_audit.len());
    for (chunked_branch, unchunked_branch) in chunked_audit.iter().zip(&unchunked_audit) {
        assert_eq!(chunked_branch.parent, unchunked_branch.parent);
        assert_eq!(
            chunked_branch.selected_index,
            unchunked_branch.selected_index
        );
        assert_eq!(chunked_branch.selected_cost, unchunked_branch.selected_cost);
        assert_eq!(
            chunked_branch.candidate_count,
            unchunked_branch.candidate_count
        );
        assert_eq!(
            chunked_branch.primitive_count,
            unchunked_branch.primitive_count
        );
        assert_eq!(
            chunked_branch.primitive_totals.len(),
            unchunked_branch.primitive_totals.len()
        );
        for (chunked_total, unchunked_total) in chunked_branch
            .primitive_totals
            .iter()
            .zip(&unchunked_branch.primitive_totals)
        {
            let tolerance = 1e-6 * chunked_total.abs().max(unchunked_total.abs()).max(1.0);
            assert!((chunked_total - unchunked_total).abs() <= tolerance);
        }
        // Packet fingerprints include padded tensor shapes, so a group feature
        // may hash differently when a different batch supplies the padding width.
        // The per-run hash remains exact and is checked again by frozen-loss recomputation.
        assert_eq!(chunked_branch.packet_sha256.len(), 64);
        assert_eq!(unchunked_branch.packet_sha256.len(), 64);
    }
    let gradient_states = chunked.gradient_exposure_states();
    assert_eq!(gradient_states[0], identity);
    assert_eq!(gradient_states[1], nonterminal);
    let mut selected_successor = nonterminal.clone();
    selected_successor
        .apply_cnot(fixture.alphabet.gates()[expected_index])
        .unwrap();
    assert_eq!(gradient_states[2], selected_successor);
    assert_eq!(gradient_states.len(), 3);
    assert_eq!(
        chunked_audit[1].candidate_count,
        fixture.alphabet.gates().len()
    );
    assert_eq!(
        chunked_audit[1].primitive_count,
        fixture.alphabet.gates().len()
    );

    let report = recompute_backup(&fixture.model, &fixture.variables, &chunked).unwrap();
    assert_eq!(report.diagnostics.examples, 2);
    assert_eq!(report.diagnostics.terminal_count, 1);
    assert_eq!(
        report.diagnostics.candidate_count,
        fixture.alphabet.gates().len()
    );
    assert!(!report.diagnostics.approximate);
    assert!(report.loss.requires_grad());
    assert!(report.loss.double_value(&[]).is_finite());

    let mut parameter = fixture.variables.trainable_variables()[0].shallow_clone();
    let replacement = Tensor::zeros_like(&parameter);
    tch::no_grad(|| parameter.f_copy_(&replacement).unwrap());
    assert!(recompute_backup(&fixture.model, &fixture.variables, &chunked).is_err());
}

#[test]
fn full_graph_objective_updates_real_model_and_tracks_actual_gradient_exposure() {
    tch::set_num_threads(1);
    tch::manual_seed(2027);
    let fixture = fixture();
    let identity = MatrixState::identity(3, fixture.alphabet.id()).unwrap();
    let gate = fixture.alphabet.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let record = TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![WordCertificate {
            start: state,
            end: identity,
            gates: vec![gate],
            physical_cost: 1,
            producer: "fixture_exact_reduction".into(),
            lineage_id: "fixture-cnot-1".into(),
        }],
        exact_distance: None,
        lineage_id: "train-fixture-1".into(),
    };
    let records = vec![record];
    let mut objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig {
            qhat_weight: 0.0,
            teacher_weight: 0.0,
            // This fixture uses deliberately synthetic providers; it tests the
            // trainer/update path, not production readiness of all 103 inputs.
            require_all_model_inputs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut runner = FullGraphObjectiveRunner::new(
        &mut objective,
        &fixture.model,
        &fixture.variables,
        &fixture.registry,
        &records,
        None,
    )
    .unwrap();
    let mut trainer = NeuralTrainer::new(
        &fixture.variables,
        Sampler::new(records.len(), 41).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    let outcome = trainer.step_with_objective(1, &mut runner).unwrap();
    assert!(outcome.accepted);
    assert!(outcome.loss.is_finite());
    drop(runner);

    let audit = objective.audit();
    assert!(audit.strict_unseen);
    assert_eq!(audit.label_count, 1);
    assert_eq!(audit.forward_count, 7);
    assert_eq!(audit.gradient_count, 2);
}

#[test]
fn full_graph_objective_checkpoint_state_is_bound_to_its_contract() {
    let fixture = fixture();
    let objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    let state = objective.checkpoint_state_bytes().unwrap();
    let contract = objective.checkpoint_contract();
    assert_eq!(
        contract["objective_sha256"].as_str(),
        Some(objective.config_sha256())
    );

    let mut restored = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    restored.restore_checkpoint_state(&state).unwrap();
    assert_eq!(restored.state_dict(), objective.state_dict());

    let mut wrong_contract: serde_json::Value = serde_json::from_slice(&state).unwrap();
    wrong_contract["config_sha256"] = serde_json::Value::String("wrong-contract".into());
    assert!(restored.load_state_dict(&wrong_contract).is_err());
}

#[test]
fn full_graph_checkpoint_contract_is_bound_to_frozen_dataset_sha256() {
    let fixture = fixture();
    let mut first = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    let unbound_digest = first.config_sha256().to_owned();
    assert!(first.bind_frozen_dataset_sha256(&"A".repeat(64)).is_err());

    let dataset_sha = "a".repeat(64);
    first.bind_frozen_dataset_sha256(&dataset_sha).unwrap();
    assert_ne!(first.config_sha256(), unbound_digest);
    assert_eq!(
        first.checkpoint_contract()["frozen_dataset_sha256"].as_str(),
        Some(dataset_sha.as_str())
    );
    first.bind_frozen_dataset_sha256(&dataset_sha).unwrap();
    assert!(first.bind_frozen_dataset_sha256(&"b".repeat(64)).is_err());

    let mut second = FullGraphObjective::new(
        vec![fixture.alphabet],
        &fixture.registry,
        fixture.policy,
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    second.bind_frozen_dataset_sha256(&"b".repeat(64)).unwrap();
    assert_ne!(first.checkpoint_contract(), second.checkpoint_contract());
}

#[test]
fn full_graph_checkpoint_contract_is_bound_to_ordered_training_records() {
    let fixture = fixture();
    let identity = MatrixState::identity(3, fixture.alphabet.id()).unwrap();
    let gate = fixture.alphabet.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let make_record = |lineage_id: &str| TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![WordCertificate {
            start: state.clone(),
            end: identity.clone(),
            gates: vec![gate],
            physical_cost: 1,
            producer: "fixture_exact_reduction".into(),
            lineage_id: lineage_id.into(),
        }],
        exact_distance: None,
        lineage_id: lineage_id.into(),
    };
    let first_records = [make_record("train-a")];
    let second_records = [make_record("train-b")];

    let mut first_objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    let dataset_sha = "c".repeat(64);
    first_objective
        .bind_frozen_dataset_sha256(&dataset_sha)
        .unwrap();
    let first_contract = {
        let runner = FullGraphObjectiveRunner::new(
            &mut first_objective,
            &fixture.model,
            &fixture.variables,
            &fixture.registry,
            &first_records,
            None,
        )
        .unwrap();
        runner.checkpoint_contract()
    };

    let mut second_objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    second_objective
        .bind_frozen_dataset_sha256(&dataset_sha)
        .unwrap();
    let second_contract = {
        let runner = FullGraphObjectiveRunner::new(
            &mut second_objective,
            &fixture.model,
            &fixture.variables,
            &fixture.registry,
            &second_records,
            None,
        )
        .unwrap();
        runner.checkpoint_contract()
    };

    assert_eq!(
        first_contract["frozen_dataset_sha256"],
        second_contract["frozen_dataset_sha256"]
    );
    assert_ne!(
        first_contract["training_records_sha256"],
        second_contract["training_records_sha256"]
    );
    assert_ne!(first_contract, second_contract);
    assert!(first_objective
        .bind_frozen_dataset_sha256(&"d".repeat(64))
        .is_err());
}

#[test]
fn full_graph_checkpoint_restores_into_fresh_trainer_and_accepts_next_update() {
    tch::set_num_threads(1);
    tch::manual_seed(2031);
    let source = fixture();
    let identity = MatrixState::identity(3, source.alphabet.id()).unwrap();
    let gate = source.alphabet.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let records = vec![TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![WordCertificate {
            start: state,
            end: identity,
            gates: vec![gate],
            physical_cost: 1,
            producer: "fixture_exact_reduction".into(),
            lineage_id: "resume-fixture-1".into(),
        }],
        exact_distance: None,
        lineage_id: "resume-train-1".into(),
    }];
    let checkpoint_dir = std::env::temp_dir().join(format!(
        "cnot103-full-graph-resume-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = CheckpointStore::new(&checkpoint_dir);

    let mut objective = FullGraphObjective::new(
        vec![source.alphabet.clone()],
        &source.registry,
        source.policy.clone(),
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
        &source.variables,
        Sampler::new(records.len(), 41).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    {
        let mut runner = FullGraphObjectiveRunner::new(
            &mut objective,
            &source.model,
            &source.variables,
            &source.registry,
            &records,
            None,
        )
        .unwrap();
        assert!(
            trainer
                .step_with_objective(1, &mut runner)
                .unwrap()
                .accepted
        );
        runner.save_checkpoint(&mut trainer, &store, false).unwrap();
    }

    let mut restored = fixture();
    let mut restored_objective = FullGraphObjective::new(
        vec![restored.alphabet.clone()],
        &restored.registry,
        restored.policy.clone(),
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
    let mut restored_trainer = NeuralTrainer::new(
        &restored.variables,
        Sampler::new(records.len(), 41).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    restored_objective
        .restore_training_checkpoint(
            &mut restored_trainer,
            &mut restored.variables,
            &store,
            &records,
        )
        .unwrap();
    assert_eq!(restored_trainer.progress().accepted, 1);
    assert_eq!(restored_trainer.optimizer_steps(), 1);

    {
        let mut runner = FullGraphObjectiveRunner::new(
            &mut restored_objective,
            &restored.model,
            &restored.variables,
            &restored.registry,
            &records,
            None,
        )
        .unwrap();
        assert!(
            restored_trainer
                .step_with_objective(1, &mut runner)
                .unwrap()
                .accepted
        );
    }
    assert_eq!(restored_trainer.progress().accepted, 2);
    fs::remove_dir_all(checkpoint_dir).unwrap();
}

#[test]
fn full_graph_objective_requires_admission_presence_to_match_checkpoint_contract() {
    let fixture = fixture();
    let identity = MatrixState::identity(3, fixture.alphabet.id()).unwrap();
    let mut state = identity.clone();
    state.apply_cnot(fixture.alphabet.gates()[0]).unwrap();
    let record = TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![WordCertificate {
            start: state,
            end: identity,
            gates: vec![fixture.alphabet.gates()[0]],
            physical_cost: 1,
            producer: "fixture_exact_reduction".into(),
            lineage_id: "fixture-cnot-1".into(),
        }],
        exact_distance: None,
        lineage_id: "train-fixture-1".into(),
    };
    let records = [record];
    let admission = AllowAdmission {
        fingerprint: "a".repeat(64),
    };
    let mut no_admission_objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        FullGraphObjectiveConfig::default(),
    )
    .unwrap();
    assert!(FullGraphObjectiveRunner::new(
        &mut no_admission_objective,
        &fixture.model,
        &fixture.variables,
        &fixture.registry,
        &records,
        Some(&admission),
    )
    .is_err());

    let mut config = FullGraphObjectiveConfig {
        external_admission: true,
        external_admission_sha256: Some(admission.fingerprint.clone()),
        ..Default::default()
    };
    let mut wrong_objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        config.clone(),
    )
    .unwrap();
    let wrong_admission = AllowAdmission {
        fingerprint: "b".repeat(64),
    };
    assert!(FullGraphObjectiveRunner::new(
        &mut wrong_objective,
        &fixture.model,
        &fixture.variables,
        &fixture.registry,
        &records,
        Some(&wrong_admission),
    )
    .is_err());

    let mut matching_objective = FullGraphObjective::new(
        vec![fixture.alphabet.clone()],
        &fixture.registry,
        fixture.policy.clone(),
        Vec::new(),
        Vec::new(),
        config.clone(),
    )
    .unwrap();
    let contract_digest = matching_objective.config_sha256().to_owned();
    let runner = FullGraphObjectiveRunner::new(
        &mut matching_objective,
        &fixture.model,
        &fixture.variables,
        &fixture.registry,
        &records,
        Some(&admission),
    )
    .unwrap();
    drop(runner);

    config.external_admission_sha256 = Some("b".repeat(64));
    let changed_objective = FullGraphObjective::new(
        vec![fixture.alphabet],
        &fixture.registry,
        fixture.policy,
        Vec::new(),
        Vec::new(),
        config,
    )
    .unwrap();
    assert_ne!(changed_objective.config_sha256(), contract_digest);
}
