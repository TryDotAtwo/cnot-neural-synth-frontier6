use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, FrequencyPlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    register_frequency_contexts, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureStatus, FrequencyContext, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'I', 7).unwrap()],
        seed: 0,
        schema_id: "cuda-frequency-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn resident(states: &[MatrixState], alphabet: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&words)
            .view([
                states.len() as i64,
                i64::from(alphabet.dimension()),
                i64::from(alphabet.dimension().div_ceil(64)),
            ])
            .to_device(Device::Cuda(0)),
        alphabet,
        0,
    )
    .unwrap()
}
fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn inputs(
    contexts: &[FrequencyContext],
    max_width: u32,
    max_frequency_bits: usize,
) -> DeviceFeatureContextInputs<'_> {
    DeviceFeatureContextInputs {
        frequency: Some(FrequencyPlanInputs {
            contexts,
            max_width,
            max_frequency_bits,
        }),
        ..Default::default()
    }
}
fn compare(
    states: &[MatrixState],
    alphabet: &CnotAlphabet,
    contexts: &[FrequencyContext],
    max_width: u32,
    max_bits: usize,
) {
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, contexts.to_vec(), max_width, max_bits).unwrap();
    let policy = policy();
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        alphabet.dimension(),
        0,
        inputs(contexts, max_width, max_bits),
    )
    .unwrap();
    let packet = plan.build(&resident(states, alphabet)).unwrap();
    let expected = registry.build(states, &policy).unwrap();
    let id = FeatureId::new(b'I', 7).unwrap();
    let actual = packet.feature(id).unwrap();
    assert_eq!(actual.values().device(), Device::Cuda(0));
    assert_eq!(flat(actual.values()), expected.feature(id).values);
    let expected_status: Vec<u8> = expected
        .feature(id)
        .status
        .iter()
        .map(|s| match s {
            FeatureStatus::Computed => 1,
            FeatureStatus::Inapplicable => 2,
            FeatureStatus::Unsupported => 3,
            FeatureStatus::ResourceLimit => 4,
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(
        Vec::<u8>::try_from(&actual.status().to_device(Device::Cpu)).unwrap(),
        expected_status
    );
}
fn matrix_cases(context: &FrequencyContext, alphabet: &CnotAlphabet) -> Vec<MatrixState> {
    let identity = MatrixState::identity(context.dimension(), alphabet.id()).unwrap();
    let s = context.sources();
    let i = context.internal();
    let t = context.targets();
    let mut valid = identity.clone();
    for (control, target) in [
        (s[0], i[0]),
        (i[i.len() - 1], t[0]),
        (s[s.len() - 1], t[t.len() - 1]),
    ] {
        valid.apply_cnot(CnotGate { control, target }).unwrap();
    }
    if i.len() > 1 {
        valid
            .apply_cnot(CnotGate {
                control: i[0],
                target: i[i.len() - 1],
            })
            .unwrap();
    }
    let mut states = vec![identity.clone(), valid];
    for (control, target) in [(i[0], s[0]), (t[0], i[0])] {
        let mut bad = identity.clone();
        bad.apply_cnot(CnotGate { control, target }).unwrap();
        states.push(bad);
    }
    if t.len() > 1 {
        let mut bad = identity;
        bad.apply_cnot(CnotGate {
            control: t[0],
            target: t[1],
        })
        .unwrap();
        states.push(bad);
    }
    states
}

#[test]
fn cuda_frequency_matches_relabelled_source_fixture_for_every_2x2_frequency() {
    let alphabet = CnotAlphabet::all_to_all_unit(5).unwrap();
    let order = [3u32, 0, 4, 1, 2];
    let canonical = [1u64, 2, 7, 13, 23];
    let mut rows = vec![0u64; 5];
    for (i, &physical_row) in order.iter().enumerate() {
        for (j, &physical_column) in order.iter().enumerate() {
            if canonical[i] & (1 << j) != 0 {
                rows[physical_row as usize] |= 1 << physical_column;
            }
        }
    }
    let states = [MatrixState::from_packed_rows(5, alphabet.id(), rows).unwrap()];
    let gpu = resident(&states, &alphabet);
    for first in 0..4u64 {
        for second in 0..4u64 {
            let contexts = vec![FrequencyContext::new(
                5,
                order[..2].to_vec(),
                order[2..3].to_vec(),
                order[3..].to_vec(),
                vec![vec![first], vec![second]],
            )
            .unwrap()];
            let mut registry = FeatureRegistry::new();
            register_frequency_contexts(&mut registry, contexts.clone(), 144, 20_736).unwrap();
            let policy = policy();
            let plan = DeviceFeaturePlan::with_contexts(
                &registry,
                &policy,
                5,
                0,
                DeviceFeatureContextInputs {
                    frequency: Some(FrequencyPlanInputs {
                        contexts: &contexts,
                        max_width: 144,
                        max_frequency_bits: 20_736,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
            let actual = plan.build(&gpu).unwrap();
            let expected = registry.build(&states, &policy).unwrap();
            let id = FeatureId::new(b'I', 7).unwrap();
            assert_eq!(
                flat(actual.feature(id).unwrap().values()),
                expected.feature(id).values
            );
            assert_eq!(actual.feature(id).unwrap().status().int64_value(&[0]), 1);
        }
    }
}

#[test]
fn cuda_frequency_matches_wide_internal_and_central_blocks_and_interface_guards() {
    for (n, r, t) in [(6, 2, 2), (65, 2, 61), (144, 2, 140), (144, 65, 4)] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let mut order: Vec<_> = (0..n).rev().collect();
        order.rotate_left(2);
        let s = order[..r].to_vec();
        let internal = order[r..r + t].to_vec();
        let targets = order[r + t..].to_vec();
        for zero in [true, false] {
            let mut central = vec![vec![0u64; r.div_ceil(64)]; targets.len()];
            if !zero {
                central[0][(r - 1) / 64] = 1 << ((r - 1) % 64);
                central[1][0] = 1;
            }
            let context =
                FrequencyContext::new(n, s.clone(), internal.clone(), targets.clone(), central)
                    .unwrap();
            compare(
                &matrix_cases(&context, &alphabet),
                &alphabet,
                &[context],
                144,
                20_736,
            );
        }
    }
}

#[test]
fn cuda_frequency_preserves_limits_alphabet_missing_width_and_metadata_identity() {
    let alphabet = CnotAlphabet::all_to_all_unit(6).unwrap();
    let c = FrequencyContext::new(
        6,
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        vec![vec![1], vec![2]],
    )
    .unwrap();
    let states = matrix_cases(&c, &alphabet);
    for (width, bits) in [(5, 4), (6, 3), (6, 4)] {
        compare(&states, &alphabet, &[c.clone()], width, bits);
    }
    let missing = FrequencyContext::new(3, vec![0], vec![1], vec![2], vec![vec![1]]).unwrap();
    compare(&states, &alphabet, &[missing], 144, 20_736);
    let wrong = CnotAlphabet::new(
        6,
        "wrong-frequency-alphabet",
        alphabet.gates().to_vec(),
        vec![1; 30],
    )
    .unwrap();
    compare(
        &[MatrixState::identity(6, wrong.id()).unwrap()],
        &wrong,
        &[c.clone()],
        5,
        3,
    );
    let contexts = [c];
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, contexts.to_vec(), 144, 20_736).unwrap();
    let policy = policy();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        6,
        0,
        DeviceFeatureContextInputs::default()
    )
    .err()
    .unwrap()
    .contains("explicit"));
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        6,
        0,
        inputs(&contexts, 143, 20_736)
    )
    .err()
    .unwrap()
    .contains("metadata"));
    let different = [FrequencyContext::new(
        6,
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        vec![vec![1], vec![1]],
    )
    .unwrap()];
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        6,
        0,
        inputs(&different, 144, 20_736)
    )
    .err()
    .unwrap()
    .contains("metadata"));
    let plan =
        DeviceFeaturePlan::with_contexts(&registry, &policy, 6, 0, inputs(&contexts, 144, 20_736))
            .unwrap();
    let gpu = resident(&states, &alphabet);
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 6, 0).unwrap();
    assert!(plan.build(&unlabelled).err().unwrap().contains("alphabet"));
    let mut relabel = policy;
    relabel.symmetry_contract = "exact_relabel".into();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &relabel,
        6,
        0,
        inputs(&contexts, 144, 20_736)
    )
    .is_err());
}

#[test]
fn cuda_frequency_packet_preserves_heads_and_named_parameter_gradients() {
    let alphabet = CnotAlphabet::all_to_all_unit(6).unwrap();
    let contexts = [FrequencyContext::new(
        6,
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        vec![vec![1], vec![2]],
    )
    .unwrap()];
    let states = matrix_cases(&contexts[0], &alphabet);
    let gpu = resident(&states, &alphabet);
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, contexts.to_vec(), 144, 20_736).unwrap();
    let mut policy = policy();
    policy.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
    let plan =
        DeviceFeaturePlan::with_contexts(&registry, &policy, 6, 0, inputs(&contexts, 144, 20_736))
            .unwrap();
    let actual = plan.build(&gpu).unwrap();
    let expected = registry.build(&states, &policy).unwrap();
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &registry,
        &policy,
        Device::Cuda(0),
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
    let a = model.forward_device(&actual).unwrap();
    let e = model.forward(&expected).unwrap();
    assert!(a.value.allclose(&e.value, 3e-5, 3e-5, false));
    for (a_values, e_values) in [(&a.qhat, &e.qhat), (&a.proposal_logits, &e.proposal_logits)] {
        assert!(a_values.masked_select(&a.action_mask).allclose(
            &e_values.masked_select(&e.action_mask),
            3e-5,
            3e-5,
            false
        ));
    }
    let loss = |x: &cnot103_native::model::FieldOutput| -> Tensor {
        x.value.mean(Kind::Float)
            + 0.01
                * x.qhat
                    .masked_select(&x.action_mask)
                    .square()
                    .mean(Kind::Float)
            + 0.01
                * x.proposal_logits
                    .masked_select(&x.action_mask)
                    .square()
                    .mean(Kind::Float)
    };
    loss(&a).backward();
    let gradients: std::collections::BTreeMap<_, _> = vars
        .variables()
        .into_iter()
        .map(|(name, v)| {
            let g = v.grad();
            (
                name,
                if g.defined() {
                    g.copy()
                } else {
                    Tensor::zeros_like(&v)
                },
            )
        })
        .collect();
    for mut v in vars.trainable_variables() {
        v.zero_grad();
    }
    loss(&e).backward();
    for (name, v) in vars.variables() {
        let g = v.grad();
        let g = if g.defined() {
            g
        } else {
            Tensor::zeros_like(&v)
        };
        assert!(gradients[&name].allclose(&g, 5e-5, 5e-5, false), "{name}");
    }
}
