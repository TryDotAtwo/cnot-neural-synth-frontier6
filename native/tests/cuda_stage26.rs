use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, Stage26PlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    register_stage26_features, register_stage26_features_with_budget, CnotAlphabet, CnotGate,
    FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState,
    STAGE26_CERTIFICATE_SOURCE_SHA256, STAGE26_CHECKER_SOURCE_SHA256,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn policy(symmetry: &str) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: [3, 4, 5].map(|i| FeatureId::new(b'F', i).unwrap()).to_vec(),
        seed: 0,
        schema_id: "cuda-stage26-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry.into(),
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
fn flat(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn inputs(limits: Stage26PlanInputs) -> DeviceFeatureContextInputs<'static> {
    DeviceFeatureContextInputs {
        stage26: Some(limits),
        ..Default::default()
    }
}
fn compare(
    states: &[MatrixState],
    alphabet: &CnotAlphabet,
    limits: Stage26PlanInputs,
    policy: &FeaturePolicy,
) {
    let mut registry = FeatureRegistry::new();
    register_stage26_features_with_budget(&mut registry, limits.max_n, limits.max_checker_pairs)
        .unwrap();
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        policy,
        alphabet.dimension(),
        0,
        inputs(limits),
    )
    .unwrap();
    let actual = plan.build(&resident(states, alphabet)).unwrap();
    let expected = registry.build(states, policy).unwrap();
    for &id in &policy.enabled_ids {
        let a = actual.feature(id).unwrap();
        let e = expected.feature(id);
        assert_eq!(
            flat(a.values()),
            e.values,
            "{id} n={}",
            alphabet.dimension()
        );
        let statuses: Vec<_> = e
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            Vec::<u8>::try_from(&a.status().to_device(Device::Cpu)).unwrap(),
            statuses
        );
    }
}

#[derive(serde::Deserialize)]
struct Golden {
    source_sha256: String,
    checker_sha256: String,
    cases: Vec<Case>,
}

#[test]
fn cuda_stage26_matches_source_on_packed_mixed_matrices_and_individual_inputs() {
    for n in [4, 65, 144] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let identity = MatrixState::identity(n, alphabet.id()).unwrap();
        let mut chain = identity.clone();
        for (control, target) in [(0, n - 1), (n - 1, 1)] {
            chain.apply_cnot(CnotGate { control, target }).unwrap();
        }
        let mut mixed = identity.clone();
        let mut rng = 0x7ac82b193u64;
        for _ in 0..4 * n {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let control = (rng % u64::from(n)) as u32;
            let target = ((rng >> 32) % u64::from(n - 1)) as u32;
            let target = if target >= control {
                target + 1
            } else {
                target
            };
            mixed.apply_cnot(CnotGate { control, target }).unwrap();
        }
        let states = [identity, chain, mixed];
        compare(
            &states,
            &alphabet,
            Stage26PlanInputs::default(),
            &policy("exact_relabel"),
        );
        if n == 4 {
            for id in [3, 4, 5] {
                let mut selected = policy("exact_relabel");
                selected.enabled_ids = vec![FeatureId::new(b'F', id).unwrap()];
                compare(&states, &alphabet, Stage26PlanInputs::default(), &selected);
            }
        }
    }
}

#[test]
fn cuda_stage26_preserves_full_pool_budgets_alphabet_and_static_identity() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states = [MatrixState::identity(3, alphabet.id()).unwrap()];
    let policy = policy("exact_relabel");
    // An identity has only three actual pool entries, but the source charges
    // choose(2*n,2)=15 pairs. Do not shrink this budget using numerical state.
    for (max_n, max_checker_pairs) in [(2, 100), (3, 14), (3, 15)] {
        compare(
            &states,
            &alphabet,
            Stage26PlanInputs {
                max_n,
                max_checker_pairs,
            },
            &policy,
        );
    }
    let wrong = CnotAlphabet::new(
        3,
        "restricted-stage26",
        alphabet.gates().to_vec(),
        vec![1; 6],
    )
    .unwrap();
    compare(
        &[MatrixState::identity(3, wrong.id()).unwrap()],
        &wrong,
        Stage26PlanInputs {
            max_n: 2,
            max_checker_pairs: 1,
        },
        &policy,
    );
    let mut registry = FeatureRegistry::new();
    register_stage26_features(&mut registry).unwrap();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        3,
        0,
        DeviceFeatureContextInputs::default()
    )
    .err()
    .unwrap()
    .contains("explicit"));
    let wrong_limits = Stage26PlanInputs {
        max_n: 143,
        ..Stage26PlanInputs::default()
    };
    assert!(
        DeviceFeaturePlan::with_contexts(&registry, &policy, 3, 0, inputs(wrong_limits))
            .err()
            .unwrap()
            .contains("metadata")
    );
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        3,
        0,
        inputs(Stage26PlanInputs::default()),
    )
    .unwrap();
    let resident = resident(&states, &alphabet);
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(resident.tensor().shallow_clone(), 3, 0).unwrap();
    assert!(plan.build(&unlabelled).err().unwrap().contains("alphabet"));
}

#[test]
fn cuda_stage26_covers_nonzero_obstruction_and_physical_relabeling() {
    // H=span(0b001110,0b110001). No input or difference of two inputs is in
    // H; its two target vectors give obstruction=2 and W0 dimension=0.
    for n in [6, 65, 144] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        for wires in [[0, 1, 2, 3, 4, 5], [0, n - 1, 1, n - 2, 2, n - 3]] {
            let mut words = MatrixState::identity(n, alphabet.id())
                .unwrap()
                .packed_rows()
                .to_vec();
            let w = n.div_ceil(64) as usize;
            for (row, columns) in [
                (wires[0], [wires[1], wires[2], wires[3]]),
                (wires[1], [wires[0], wires[4], wires[5]]),
            ] {
                words[row as usize * w..(row as usize + 1) * w].fill(0);
                for col in columns {
                    words[row as usize * w + col as usize / 64] |= 1u64 << (col % 64);
                }
            }
            let state = MatrixState::from_packed_rows(n, alphabet.id(), words).unwrap();
            compare(
                &[state.clone()],
                &alphabet,
                Stage26PlanInputs::default(),
                &policy("exact_relabel"),
            );
            let mut registry = FeatureRegistry::new();
            register_stage26_features(&mut registry).unwrap();
            let packet = registry.build(&[state], &policy("exact_relabel")).unwrap();
            assert_eq!(
                packet.feature(FeatureId::new(b'F', 3).unwrap()).values,
                vec![0.0, 2.0, 2.0, 2.0, 0.0]
            );
            assert_eq!(
                packet.feature(FeatureId::new(b'F', 5).unwrap()).values,
                vec![4.0, -2.0]
            );
        }
    }
}

#[test]
fn cuda_stage26_packet_preserves_model_heads_and_parameter_gradients() {
    let n = 4;
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let states = [
        MatrixState::identity(n, alphabet.id()).unwrap(),
        MatrixState::from_packed_rows(n, alphabet.id(), vec![1, 3, 7, 8]).unwrap(),
        MatrixState::from_packed_rows(n, alphabet.id(), vec![2, 1, 4, 8]).unwrap(),
    ];
    let mut policy = policy("exact_relabel");
    policy.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
    let mut registry = FeatureRegistry::new();
    register_stage26_features(&mut registry).unwrap();
    let gpu = resident(&states, &alphabet);
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        n,
        0,
        inputs(Stage26PlanInputs::default()),
    )
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
    for (actual, expected) in [(&a.qhat, &e.qhat), (&a.proposal_logits, &e.proposal_logits)] {
        assert!(actual.masked_select(&a.action_mask).allclose(
            &expected.masked_select(&e.action_mask),
            3e-5,
            3e-5,
            false
        ));
    }
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
#[derive(serde::Deserialize)]
struct Case {
    rows: Vec<u64>,
    distance: usize,
    f03: Vec<f32>,
    f04: Vec<f32>,
    f05: Vec<f32>,
}

#[test]
fn cuda_stage26_matches_all_gl3_source_golden_and_distance_bounds() {
    let golden: Golden =
        serde_json::from_str(include_str!("fixtures/stage26_gl3_source_golden.json")).unwrap();
    assert_eq!(golden.source_sha256, STAGE26_CERTIFICATE_SOURCE_SHA256);
    assert_eq!(golden.checker_sha256, STAGE26_CHECKER_SOURCE_SHA256);
    assert_eq!(golden.cases.len(), 168);
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states: Vec<_> = golden
        .cases
        .iter()
        .map(|c| MatrixState::from_packed_rows(3, alphabet.id(), c.rows.clone()).unwrap())
        .collect();
    let gpu = resident(&states, &alphabet);
    let mut registry = FeatureRegistry::new();
    register_stage26_features(&mut registry).unwrap();
    for symmetry in ["exact_relabel", "augmentation_only"] {
        let policy = policy(symmetry);
        let plan = DeviceFeaturePlan::with_contexts(
            &registry,
            &policy,
            3,
            0,
            DeviceFeatureContextInputs {
                stage26: Some(Stage26PlanInputs::default()),
                ..Default::default()
            },
        )
        .unwrap();
        let packet = plan.build(&gpu).unwrap();
        for (b, case) in golden.cases.iter().enumerate() {
            for (id, expected) in [(3, &case.f03), (4, &case.f04), (5, &case.f05)] {
                let actual = packet.feature(FeatureId::new(b'F', id).unwrap()).unwrap();
                assert_eq!(actual.values().device(), Device::Cuda(0));
                assert_eq!(actual.status().int64_value(&[b as i64]), 1);
                assert_eq!(
                    &flat(&actual.values().get(b as i64)),
                    expected,
                    "F{id:02} at {:?}",
                    case.rows
                );
            }
            assert!(case.f05[0] <= case.distance as f32);
        }
    }
}
