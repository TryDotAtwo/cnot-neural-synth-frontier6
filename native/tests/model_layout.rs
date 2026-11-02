use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::nn_training::{AdamWConfig, NeuralTrainer};
use cnot103_native::packet::FeaturePacket;
use cnot103_native::{
    all_feature_specs, CnotAlphabet, CnotGate, FeatureComputation, FeatureId, FeaturePolicy,
    FeatureProducer, FeatureRegistry, FeatureRole, FeatureSpec, FeatureValue, MatrixState, Sampler,
};
use std::collections::BTreeMap;
use tch::nn::VarStore;
use tch::Device;

struct DynamicPairFeature(FeatureSpec);

impl FeatureProducer for DynamicPairFeature {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        let density = state
            .packed_rows()
            .iter()
            .map(|row| row.count_ones())
            .sum::<u32>() as f32;
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, n, 3],
            values: vec![density; n * n * 3],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

fn dynamic_model_fixture(
    device: Device,
) -> (
    VarStore,
    FusionDistanceField,
    FeatureRegistry,
    FeaturePolicy,
    FeaturePacket,
) {
    let id = FeatureId::new(b'I', 11).unwrap();
    let mut spec = all_feature_specs()[id.index()].clone();
    spec.role = FeatureRole::Pair;
    spec.channels = 3;
    spec.wire_roles.clear();
    spec = spec.with_provider_metadata(
        "full_labeled_matrix",
        "fixture_i11_pair_v1",
        ["M"],
        "exact",
        "T1",
        std::iter::empty::<&str>(),
        false,
    );

    let mut registry = FeatureRegistry::new();
    registry
        .register(Box::new(DynamicPairFeature(spec)))
        .unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let mut state = MatrixState::identity(2, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "dynamic-model-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&[state], &policy).unwrap();
    let variables = VarStore::new(device);
    let model = FusionDistanceField::new_for_policy(
        &variables.root(),
        &registry,
        &policy,
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

    (variables, model, registry, policy, packet)
}

#[test]
fn model_uses_only_selected_features_with_their_registered_shapes() {
    let (variables, model, _registry, _policy, packet) = dynamic_model_fixture(Device::Cpu);

    let output = model.forward(&packet).unwrap();
    assert_eq!(output.value.size(), [1]);
    assert_eq!(output.proposal_logits.size(), [1, 2, 2]);
    assert_eq!(output.qhat.size(), [1, 2, 2]);
    assert!(output.value.isfinite().all().int64_value(&[]) != 0);
    assert_eq!(output.action_mask.sum(tch::Kind::Int64).int64_value(&[]), 2);
    let valid_logits = output.proposal_logits.masked_select(&output.action_mask);
    let invalid_logits = output
        .proposal_logits
        .masked_select(&output.action_mask.logical_not());
    let valid_qhat = output.qhat.masked_select(&output.action_mask);
    let invalid_qhat = output.qhat.masked_select(&output.action_mask.logical_not());
    assert!(valid_logits.isfinite().all().int64_value(&[]) != 0);
    assert!(invalid_logits.eq(f64::NEG_INFINITY).all().int64_value(&[]) != 0);
    assert!(valid_qhat.isfinite().all().int64_value(&[]) != 0);
    assert!(invalid_qhat.eq(f64::INFINITY).all().int64_value(&[]) != 0);

    let variable_names = variables.variables();
    for name in [
        "stem.projections.I11.weight",
        "stem.raw_matrix.weight",
        "stem.pair_to_node.weight",
        "stem.size_projection.weight",
        "stem.context_q.weight",
        "stem.blocks.0.qkv.weight",
        "family_heads.heads.I.0.weight",
        "fusion.family_type",
        "fusion.family_attention.self_attn.in_proj_weight",
        "fusion.family_attention.linear1.weight",
        "fusion.blocks.0.qkv.weight",
        "fusion.dist",
        "fusion.read.in_proj_weight",
        "fusion.norm.weight",
        "fusion.output.0.weight",
        "auxiliary.proposal.0.weight",
    ] {
        assert!(
            variable_names.contains_key(name),
            "reference state_dict parameter name missing: {name}"
        );
    }

    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/python_model_state_dict_shapes.json")).unwrap();
    let expected: BTreeMap<String, Vec<i64>> =
        serde_json::from_value(reference["state_dict"].clone()).unwrap();
    let actual: BTreeMap<_, _> = variables
        .variables()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.size()))
        .collect();
    assert_eq!(
        actual, expected,
        "native model parameter names and shapes must match the frozen Python reference"
    );

    let initial = variables.variables();
    let maximum_abs = |name: &str| initial[name].abs().max().double_value(&[]);
    let linear_bound = 1.0 / 8.0f64.sqrt();
    let fused_attention_bound = (1.5 / 8.0f64).sqrt();
    assert!(
        maximum_abs("fusion.family_attention.linear1.weight") <= linear_bound,
        "TransformerEncoderLayer linear1 must use PyTorch nn.Linear initialization"
    );
    assert!(
        maximum_abs("fusion.family_attention.linear2.weight") <= 1.0 / 16.0f64.sqrt(),
        "TransformerEncoderLayer linear2 must use PyTorch nn.Linear initialization"
    );
    assert!(
        maximum_abs("fusion.family_attention.self_attn.out_proj.weight") <= linear_bound,
        "MultiheadAttention out_proj must retain PyTorch nn.Linear initialization"
    );
    assert!(
        maximum_abs("fusion.family_attention.self_attn.in_proj_weight") <= fused_attention_bound,
        "MultiheadAttention fused in_proj_weight must use PyTorch Xavier initialization"
    );
    assert_eq!(
        initial["fusion.family_attention.self_attn.out_proj.bias"]
            .abs()
            .max()
            .double_value(&[]),
        0.0,
        "MultiheadAttention resets the out_proj bias to zero"
    );
}

#[test]
fn default_builtin_feature_set_executes_through_model_and_adamw_on_cuda() {
    tch::set_num_threads(1);
    cnot103_native::torch_device::require_cuda_device(0)
        .expect("initialize CUDA for the native-feature integration test");
    let device = Device::Cuda(0);

    let registry = FeatureRegistry::new();
    let mut policy = FeaturePolicy::complete_103("native-default-feature-set-cuda-v1");
    let candidate_ids = policy.enabled_ids.clone();
    let selected_ids: Vec<_> = candidate_ids
        .into_iter()
        .filter(|id| {
            let mut probe = policy.clone();
            probe.enabled_ids = vec![*id];
            registry.selected_specs(&probe).is_ok()
        })
        .collect();
    policy.enabled_ids = selected_ids;
    let specs = registry.selected_specs(&policy).unwrap();
    assert_eq!(specs.len(), 41, "the default registry exposes the 41 built-in feature producers");

    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let packet = registry.build(&[state], &policy).unwrap();

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
    let output = model.forward(&packet).unwrap();
    assert!(output.value.isfinite().all().int64_value(&[]) != 0);
    assert_eq!(output.value.device(), device);

    let mut trainer = NeuralTrainer::new(
        &variables,
        Sampler::new(1, 7).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    let outcome = trainer
        .step(1, |_| Ok(output.value.sum(tch::Kind::Float)))
        .unwrap();
    assert!(outcome.accepted);
    assert_eq!(trainer.progress().accepted, 1);
    assert_eq!(trainer.optimizer_steps(), 1);
}

#[test]
fn model_forward_and_adamw_update_execute_on_cuda_when_available() {
    tch::set_num_threads(1);
    if let Err(error) = cnot103_native::torch_device::require_cuda_device(0) {
        assert!(
            std::env::var_os("CNOT103_REQUIRE_CUDA").is_none(),
            "CNOT103_REQUIRE_CUDA is set, but CUDA initialization failed: {error}"
        );
        eprintln!("CUDA unavailable; skipping the CUDA model update smoke test: {error}");
        return;
    }

    let device = Device::Cuda(0);
    let (variables, model, _registry, _policy, packet) = dynamic_model_fixture(device);
    let output = model.forward(&packet).unwrap();
    assert_eq!(output.value.device(), device);
    assert!(output.value.isfinite().all().int64_value(&[]) != 0);

    let before: BTreeMap<_, _> = variables
        .variables()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.copy()))
        .collect();
    let loss = output.value.sum(tch::Kind::Float);
    let mut trainer = NeuralTrainer::new(
        &variables,
        Sampler::new(1, 0).unwrap(),
        AdamWConfig::default(),
    )
    .unwrap();
    let outcome = trainer.step(1, |_| Ok(loss)).unwrap();
    assert!(
        outcome.accepted,
        "CUDA model loss must produce an admitted step"
    );
    assert_eq!(trainer.progress().attempted, 1);
    assert_eq!(trainer.progress().accepted, 1);
    assert_eq!(trainer.optimizer_steps(), 1);

    let after = variables.variables();
    assert!(before
        .iter()
        .any(|(name, value)| !value.equal(&after[name])));
}
