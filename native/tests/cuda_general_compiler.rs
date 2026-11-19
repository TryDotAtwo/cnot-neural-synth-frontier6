use cnot103_native::device_compiler::{verify_device_cnot_words, DeviceGeneralCompilerPlan};
use cnot103_native::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    compile_general_cnot_word, register_general_compiler_bounds, CnotAlphabet, CnotGate, FeatureId,
    FeaturePolicy, FeatureRegistry, GeneralCompilerError, GeneralCompilerLimits, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn resident(states: &[MatrixState], alphabet: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&w| w as i64))
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
fn flat(x: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&x.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn check_words(states: &[MatrixState], alphabet: &CnotAlphabet, limits: GeneralCompilerLimits) {
    let gpu = resident(states, alphabet);
    let plan = DeviceGeneralCompilerPlan::new(alphabet.dimension(), 0, limits).unwrap();
    let words = plan.compile(&gpu).unwrap();
    assert_eq!(words.gates().device(), Device::Cuda(0));
    let status = Vec::<u8>::try_from(&words.status().to_device(Device::Cpu)).unwrap();
    let lengths = Vec::<i64>::try_from(&words.lengths().to_device(Device::Cpu)).unwrap();
    let lower = Vec::<i64>::try_from(&words.lower_bounds().to_device(Device::Cpu)).unwrap();
    for (b, state) in states.iter().enumerate() {
        let actual = Vec::<i32>::try_from(
            &words
                .gates()
                .get(b as i64)
                .to_device(Device::Cpu)
                .view([-1]),
        )
        .unwrap();
        match compile_general_cnot_word(state, limits) {
            Ok(expected) => {
                assert_eq!(status[b], 1);
                assert_eq!(lengths[b], expected.gates.len() as i64);
                let expected_gates: Vec<i32> = expected
                    .gates
                    .iter()
                    .flat_map(|g| [g.control as i32, g.target as i32])
                    .collect();
                assert_eq!(&actual[..expected_gates.len()], expected_gates.as_slice());
                assert!(actual[expected_gates.len()..].iter().all(|&v| v == 0));
                let mut replay = MatrixState::identity(state.dimension(), alphabet.id()).unwrap();
                for pair in actual[..expected_gates.len()].chunks_exact(2) {
                    replay
                        .apply_cnot(CnotGate {
                            control: pair[0] as u32,
                            target: pair[1] as u32,
                        })
                        .unwrap();
                }
                assert_eq!(&replay, state, "independent replay b={b}");
                assert_eq!(
                    lower[b],
                    cnot103_native::training_data::elementary_lower_bound(state).unwrap() as i64
                );
                let tight = f32::from(lengths[b] == lower[b]);
                assert_eq!(
                    flat(&words.f09().get(b as i64)),
                    vec![
                        lengths[b] as f32,
                        lower[b] as f32,
                        (lengths[b] - lower[b]) as f32,
                        tight
                    ]
                );
                assert_eq!(
                    flat(&words.f10().get(b as i64)),
                    vec![1.0, 1.0, tight, 0.0, 1.0]
                );
                assert_eq!(
                    words.f10().double_value(&[b as i64, 3]),
                    0.0,
                    "no exhaustive search claim"
                );
            }
            Err(error) => {
                let expected = match error {
                    GeneralCompilerError::ResourceLimit(_) => 4,
                    _ => 3,
                };
                assert_eq!(status[b], expected);
                assert_eq!(lengths[b], 0);
                assert_eq!(lower[b], 0);
                assert!(actual.iter().all(|&v| v == 0));
                assert!(flat(&words.f09().get(b as i64)).iter().all(|&v| v == 0.0));
                assert!(flat(&words.f10().get(b as i64)).iter().all(|&v| v == 0.0));
            }
        }
    }
}

#[test]
fn cuda_general_compiler_matches_all_gl3_words_and_bounds() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states: Vec<_> = (0u16..512)
        .filter_map(|bits| {
            MatrixState::from_packed_rows(
                3,
                alphabet.id(),
                (0..3)
                    .map(|row| u64::from((bits >> (row * 3)) & 7))
                    .collect(),
            )
            .ok()
        })
        .collect();
    assert_eq!(states.len(), 168);
    check_words(&states, &alphabet, GeneralCompilerLimits::default());
}

#[test]
fn cuda_general_compiler_preserves_gate_limits_and_zero_capacity() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let states = [
        MatrixState::identity(2, alphabet.id()).unwrap(),
        MatrixState::from_packed_rows(2, alphabet.id(), vec![2, 1]).unwrap(),
    ];
    for max_gates in [0, 2, 3] {
        check_words(
            &states,
            &alphabet,
            GeneralCompilerLimits {
                max_dimension: 2,
                max_gates,
            },
        );
    }
    check_words(
        &states,
        &alphabet,
        GeneralCompilerLimits {
            max_dimension: 1,
            max_gates: 100,
        },
    );
}

#[test]
fn cuda_general_compiler_covers_packed_widths_and_default_maximum() {
    for n in [65, 144, 1024] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let identity = MatrixState::identity(n, alphabet.id()).unwrap();
        let mut changed = identity.clone();
        for (control, target) in [(n - 1, 0), (0, n - 1), (n - 1, 0), (1, n - 2)] {
            changed.apply_cnot(CnotGate { control, target }).unwrap();
        }
        let mut mixed = identity.clone();
        if n < 1024 {
            for step in 0..3 * n {
                let control = (17 * step + 1) % n;
                let mut target = (29 * step + 2) % n;
                if control == target {
                    target = (target + 1) % n;
                }
                mixed.apply_cnot(CnotGate { control, target }).unwrap();
            }
        }
        check_words(
            &[identity, changed, mixed],
            &alphabet,
            GeneralCompilerLimits::default(),
        );
    }
}

#[test]
fn cuda_general_word_verifier_rejects_corrupted_gates_and_lengths() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::from_packed_rows(3, alphabet.id(), vec![1, 3, 7]).unwrap();
    let gpu = resident(&[state], &alphabet);
    let words = DeviceGeneralCompilerPlan::new(3, 0, GeneralCompilerLimits::default())
        .unwrap()
        .compile(&gpu)
        .unwrap();
    let verify = |gates: &Tensor, lengths: &Tensor| {
        verify_device_cnot_words(&gpu, gates, lengths)
            .unwrap()
            .int64_value(&[0])
    };
    assert_eq!(verify(words.gates(), words.lengths()), 1);
    let gates = words.gates().copy();
    let _ = gates.get(0).get(0).fill_(0);
    assert_eq!(verify(&gates, words.lengths()), 0);
    let _ = gates.get(0).get(0).get(0).fill_(3);
    assert_eq!(verify(&gates, words.lengths()), 0);
    assert_eq!(verify(words.gates(), &words.lengths().full_like(-1)), 0);
    assert_eq!(
        verify(
            words.gates(),
            &words.lengths().full_like(words.gates().size()[1] + 1)
        ),
        0
    );
}

#[test]
fn cuda_compiler_bound_packet_preserves_source_heads_and_gradients() {
    let n = 4;
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let mut state = MatrixState::identity(n, alphabet.id()).unwrap();
    for (control, target) in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 3)] {
        state.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [MatrixState::identity(n, alphabet.id()).unwrap(), state];
    let gpu = resident(&states, &alphabet);
    let mut registry = FeatureRegistry::new();
    register_general_compiler_bounds(&mut registry).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: [(b'A', 1), (b'F', 9), (b'F', 10)]
            .map(|(f, i)| FeatureId::new(f, i).unwrap())
            .to_vec(),
        seed: 0,
        schema_id: "cuda-general-bounds-v1".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        n,
        0,
        DeviceFeatureContextInputs::default(),
    )
    .unwrap();
    let actual = plan.build(&gpu).unwrap();
    let expected = registry.build(&states, &policy).unwrap();
    for &id in &policy.enabled_ids {
        assert_eq!(
            flat(actual.feature(id).unwrap().values()),
            expected.feature(id).values
        );
    }
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
    assert!(a.qhat.masked_select(&a.action_mask).allclose(
        &e.qhat.masked_select(&e.action_mask),
        3e-5,
        3e-5,
        false
    ));
    assert!(a.proposal_logits.masked_select(&a.action_mask).allclose(
        &e.proposal_logits.masked_select(&e.action_mask),
        3e-5,
        3e-5,
        false
    ));
    let loss = |x: &cnot103_native::model::FieldOutput| {
        x.value.mean(Kind::Float)
            + x.qhat
                .masked_select(&x.action_mask)
                .square()
                .mean(Kind::Float)
                * 0.01
            + x.proposal_logits
                .masked_select(&x.action_mask)
                .square()
                .mean(Kind::Float)
                * 0.01
    };
    loss(&a).backward();
    let gradients = vars
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
        .collect::<std::collections::BTreeMap<_, _>>();
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
    let mut disabled = policy.clone();
    disabled.solver_assisted = false;
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &disabled,
        n,
        0,
        DeviceFeatureContextInputs::default()
    )
    .is_err());
}

#[test]
fn cuda_general_compiler_requires_an_explicit_matching_alphabet() {
    let canonical = CnotAlphabet::all_to_all_unit(3).unwrap();
    let wrong = CnotAlphabet::new(
        3,
        "different-alphabet",
        canonical.gates().to_vec(),
        vec![1; 6],
    )
    .unwrap();
    let state = MatrixState::identity(3, wrong.id()).unwrap();
    let gpu = resident(&[state], &wrong);
    let plan = DeviceGeneralCompilerPlan::new(3, 0, GeneralCompilerLimits::default()).unwrap();
    let words = plan.compile(&gpu).unwrap();
    assert_eq!(words.status().int64_value(&[0]), 3);
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 3, 0).unwrap();
    assert!(plan.compile(&unlabelled).unwrap_err().contains("alphabet"));
}
