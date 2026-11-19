use super::*;
use cnot103_native::nn_training::{AdamWConfig, NeuralTrainer};
use cnot103_native::objective::{
    FullGraphObjective, FullGraphObjectiveConfig, FullGraphObjectiveRunner,
};
use cnot103_native::training_data::{TrainingRecord, TrainingSplit};
use cnot103_native::{Sampler, WordCertificate};
use std::collections::HashMap;

pub(super) fn compare_updates(
    plan: DeviceFeaturePlan,
    sampled_plan: DeviceFeaturePlan,
    micro_plan: DeviceFeaturePlan,
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    sampled_policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    states: &[MatrixState; 2],
) {
    let device = Device::Cuda(0);
    let reduction = vec![
        CnotGate {
            control: 1,
            target: 3,
        },
        CnotGate {
            control: 0,
            target: 3,
        },
        CnotGate {
            control: 0,
            target: 2,
        },
    ];
    let alternative: Vec<_> = [alphabet.gates()[0]; 2]
        .into_iter()
        .chain(reduction.iter().copied())
        .collect();
    let word = |s: &MatrixState, gates: Vec<CnotGate>| WordCertificate {
        start: s.clone(),
        end: states[0].clone(),
        physical_cost: gates.len() as u64,
        gates,
        producer: "independent-physical-replay".into(),
        lineage_id: "admitted-test-record".into(),
    };
    let records = vec![
        TrainingRecord {
            state: states[0].clone(),
            split: TrainingSplit::Train,
            verified_words: vec![word(&states[0], vec![])],
            exact_distance: Some(0),
            lineage_id: "terminal".into(),
        },
        TrainingRecord {
            state: states[1].clone(),
            split: TrainingSplit::Train,
            verified_words: vec![word(&states[1], reduction), word(&states[1], alternative)],
            exact_distance: None,
            lineage_id: "two-teachers".into(),
        },
    ];
    let config = ModelConfig {
        wire_dim: 8,
        pair_dim: 4,
        structural_blocks: 1,
        attention_heads: 2,
        feedforward_dim: 16,
        fusion_blocks: 1,
        basis_proposal_hidden_dim: 8,
    };
    tch::manual_seed(99781);
    let source_vars = VarStore::new(device);
    let mut gpu_vars = VarStore::new(device);
    let source_model = FusionDistanceField::new_for_policy(
        &source_vars.root(),
        registry,
        policy,
        device,
        config.clone(),
    )
    .unwrap();
    let gpu_model = FusionDistanceField::new_for_policy(
        &gpu_vars.root(),
        registry,
        policy,
        device,
        config.clone(),
    )
    .unwrap();
    gpu_vars.copy(&source_vars).unwrap();
    let objective = || {
        FullGraphObjective::new(
            vec![alphabet.clone()],
            registry,
            policy.clone(),
            vec![],
            vec![],
            FullGraphObjectiveConfig {
                bellman_weight: 0.75,
                supervision_weight: 0.6,
                qhat_weight: 0.17,
                teacher_weight: 0.43,
                teacher_temperature: 0.7,
                neighbor_chunk_size: 5,
                require_all_model_inputs: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let trainer = |v: &VarStore| {
        NeuralTrainer::new(
            v,
            Sampler::new(records.len(), 7).unwrap(),
            AdamWConfig {
                learning_rate: 2e-4,
                weight_decay: 0.2,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let (mut reference, mut gpu) = (objective(), objective());
    let (mut ref_train, mut gpu_train) = (trainer(&source_vars), trainer(&gpu_vars));
    let mut ref_runner = FullGraphObjectiveRunner::new(
        &mut reference,
        &source_model,
        &source_vars,
        registry,
        &records,
        None,
    )
    .unwrap();
    let mut runner =
        FullGraphObjectiveRunner::new(&mut gpu, &gpu_model, &gpu_vars, registry, &records, None)
            .unwrap()
            .into_device(HashMap::from([(4, plan)]))
            .unwrap();
    for _ in 0..2 {
        let r = ref_train.step_with_objective(2, &mut ref_runner).unwrap();
        let g = runner.run_interval(&mut gpu_train, 1, 2).unwrap();
        assert!(r.accepted && g.accepted);
        assert!(
            (r.loss - g.loss).abs() < 3e-5,
            "GPU {} vs source {}",
            g.loss,
            r.loss
        );
    }
    drop(runner);
    drop(ref_runner);
    assert_eq!(reference.audit(), gpu.audit());
    assert_eq!(reference.diagnostics(), gpu.diagnostics());
    assert_eq!(gpu.diagnostics().exact_examples, 1);
    assert_eq!(gpu.diagnostics().teacher_examples, 1);
    assert_eq!(gpu.diagnostics().primitive_candidates, 12);
    assert_eq!(reference.state_dict(), gpu.state_dict());
    assert_eq!(gpu_train.sampler_snapshot(), ref_train.sampler_snapshot());
    let (meta, moments) =
        super::optimizer_snapshot::saved(&gpu_vars, &gpu_train, "full-graph-device");
    let (expected, reference_moments) =
        super::optimizer_snapshot::saved(&source_vars, &ref_train, "full-graph-reference");
    assert_eq!(
        meta, expected,
        "all named parameter step counts and signatures"
    );
    let source = source_vars.variables();
    let mut max_weight_error = 0.0f64;
    let mut max_moment_error = 0.0f64;
    for (name, t) in gpu_vars.variables() {
        let r = &source[&name];
        max_weight_error = max_weight_error.max((&t - r).abs().max().double_value(&[]));
        assert!(
            t.allclose(r, 2e-6, 2e-6, false),
            "{name} weight err {}",
            (&t - r).abs().max().double_value(&[])
        );
    }
    for (name, t) in &moments {
        let r = &reference_moments[name];
        max_moment_error = max_moment_error.max((t - r).abs().max().double_value(&[]));
        assert!(
            t.allclose(r, 1e-5, 2e-7, false),
            "{name} moment err {}",
            (t - r).abs().max().double_value(&[])
        );
    }
    println!("source objective parity inputs=102 records=2 steps=2 parameters={} moments={} max_weight_abs={} max_moment_abs={} gradient_states={}",
        source.len(),moments.len(),max_weight_error,max_moment_error,gpu.audit().gradient_count);

    // Exercise the actual sampled Bellman objective and all 102 inputs using
    // the same logical batches, initial weights, targets, and optimizer state.
    // A batch of four intentionally crosses the two-record epoch boundary.
    let sampled_vars = VarStore::new(device);
    let mut micro_vars = VarStore::new(device);
    let sampled_model = FusionDistanceField::new_for_policy(
        &sampled_vars.root(),
        registry,
        sampled_policy,
        device,
        config.clone(),
    )
    .unwrap();
    let micro_model = FusionDistanceField::new_for_policy(
        &micro_vars.root(),
        registry,
        sampled_policy,
        device,
        config,
    )
    .unwrap();
    micro_vars.copy(&sampled_vars).unwrap();
    let sampled_objective = |sequential| {
        FullGraphObjective::new(
            vec![alphabet.clone()],
            registry,
            sampled_policy.clone(),
            vec![],
            vec![],
            FullGraphObjectiveConfig {
                bellman_weight: 0.75,
                supervision_weight: 0.6,
                qhat_weight: 0.17,
                teacher_weight: 0.43,
                teacher_temperature: 0.7,
                neighbor_chunk_size: 5,
                sampled_neighbor_count: Some(4),
                sequential_batch_gradients: sequential,
                require_all_model_inputs: true,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let (mut sampled, mut micro) = (sampled_objective(false), sampled_objective(true));
    let (mut sampled_trainer, mut micro_trainer) = (trainer(&sampled_vars), trainer(&micro_vars));
    let mut sampled_runner = FullGraphObjectiveRunner::new(
        &mut sampled,
        &sampled_model,
        &sampled_vars,
        registry,
        &records,
        None,
    )
    .unwrap()
    .into_device(HashMap::from([(4, sampled_plan)]))
    .unwrap();
    let mut micro_runner = FullGraphObjectiveRunner::new(
        &mut micro,
        &micro_model,
        &micro_vars,
        registry,
        &records,
        None,
    )
    .unwrap()
    .into_device(HashMap::from([(4, micro_plan)]))
    .unwrap();
    for _ in 0..2 {
        let a = sampled_runner
            .run_interval(&mut sampled_trainer, 1, 4)
            .unwrap();
        let b = micro_runner.run_interval(&mut micro_trainer, 1, 4).unwrap();
        assert!(a.accepted && b.accepted);
        assert!(
            (a.loss - b.loss).abs() < 5e-5,
            "sampled {} vs micro {}",
            a.loss,
            b.loss
        );
    }
    drop(sampled_runner);
    drop(micro_runner);
    assert_eq!(sampled.audit(), micro.audit());
    assert_eq!(sampled.diagnostics(), micro.diagnostics());
    assert_eq!(sampled.diagnostics().examples, 4);
    assert_eq!(sampled.diagnostics().primitive_candidates, 8);
    assert_eq!(
        sampled_trainer.sampler_snapshot(),
        micro_trainer.sampler_snapshot()
    );
    let mut a = sampled.state_dict();
    let mut b = micro.state_dict();
    a.as_object_mut().unwrap().remove("config_sha256");
    b.as_object_mut().unwrap().remove("config_sha256");
    assert_eq!(a, b, "sampled candidate and gradient exposure ledger");
    let (meta_a, moments_a) =
        super::optimizer_snapshot::saved(&sampled_vars, &sampled_trainer, "sampled-full-batch");
    let (meta_b, moments_b) =
        super::optimizer_snapshot::saved(&micro_vars, &micro_trainer, "sampled-micro-batch");
    assert_eq!(meta_a, meta_b);
    let expected = sampled_vars.variables();
    for (name, actual) in micro_vars.variables() {
        assert!(
            actual.allclose(&expected[&name], 3e-6, 3e-6, false),
            "sampled microbatch weight {name} error {}",
            (&actual - &expected[&name]).abs().max().double_value(&[])
        );
    }
    for (name, actual) in &moments_b {
        assert!(
            actual.allclose(&moments_a[name], 5e-5, 2e-7, false),
            "sampled microbatch moment {name} error {}",
            (actual - &moments_a[name]).abs().max().double_value(&[])
        );
    }
    println!("sampled 102-input full-objective microbatch parity: records=2 batch=4 updates=2");
}
