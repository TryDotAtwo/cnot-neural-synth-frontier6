use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, SharedRelationPlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    register_shared_relation_contexts, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureStatus, MatrixState, SharedRelationContext,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'I', 6).unwrap()],
        seed: 0,
        schema_id: "cuda-shared-relation-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn resident(states: &[MatrixState]) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = states[0].dimension();
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect();
    DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&words)
            .view([states.len() as i64, i64::from(n), i64::from(n.div_ceil(64))])
            .to_device(Device::Cuda(0)),
        n,
        0,
    )
    .unwrap()
}
fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn inputs(
    contexts: &[SharedRelationContext],
    limits: [usize; 3],
) -> DeviceFeatureContextInputs<'_> {
    DeviceFeatureContextInputs {
        shared_relation: Some(SharedRelationPlanInputs {
            contexts,
            max_vectors: limits[0],
            max_coordinates: limits[1],
            max_ports: limits[2],
        }),
        ..Default::default()
    }
}
fn compare(states: &[MatrixState], contexts: &[SharedRelationContext], limits: [usize; 3]) {
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(
        &mut registry,
        contexts.to_vec(),
        limits[0],
        limits[1],
        limits[2],
    )
    .unwrap();
    let policy = policy();
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        states[0].dimension(),
        0,
        inputs(contexts, limits),
    )
    .unwrap();
    let packet = plan.build(&resident(states)).unwrap();
    let expected = registry.build(states, &policy).unwrap();
    let id = FeatureId::new(b'I', 6).unwrap();
    let a = packet.feature(id).unwrap();
    let e = expected.feature(id);
    let g = e.shape[1] as i64;
    let capacity = a.values().size()[1];
    assert_eq!(a.values().device(), Device::Cuda(0));
    assert!(capacity >= g);
    assert_eq!(flat(&a.values().narrow(1, 0, g)), e.values);
    assert_eq!(
        flat(&a.incidence().unwrap().narrow(1, 0, g)),
        *e.incidence.as_ref().unwrap()
    );
    let mask = a
        .token_valid()
        .unwrap()
        .narrow(1, 0, g)
        .to_kind(Kind::Uint8)
        .to_device(Device::Cpu)
        .contiguous()
        .view([-1]);
    assert_eq!(
        Vec::<u8>::try_from(&mask).unwrap(),
        *e.token_valid.as_ref().unwrap()
    );
    for payload in [a.values(), a.incidence().unwrap(), a.token_valid().unwrap()] {
        assert!(
            flat(&payload.narrow(1, g, capacity - g).to_kind(Kind::Float))
                .iter()
                .all(|&v| v == 0.0)
        );
    }
    let statuses: Vec<u8> = e
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
#[test]
fn cuda_shared_relation_matches_full_source_kernel_ports_and_incidence_for_all_gl3() {
    let states: Vec<_> = (0..512u64)
        .filter_map(|bits| {
            MatrixState::from_packed_rows(
                3,
                "i06-custom-alphabet",
                (0..3).map(|row| (bits >> (3 * row)) & 7).collect(),
            )
            .ok()
        })
        .collect();
    assert_eq!(states.len(), 168);
    let contexts =
        vec![
            SharedRelationContext::new(3, vec![2, 0, 1], vec![1, 0], vec![vec![1], vec![2]])
                .unwrap(),
        ];
    compare(&states, &contexts, [3, 2, 2]);
}

fn swaps(n: u32) -> Vec<MatrixState> {
    let identity = MatrixState::identity(n, "shared-relation-custom").unwrap();
    let mut a = identity.clone();
    for (control, target) in [(0, n - 1), (n - 1, 0), (0, n - 1)] {
        a.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let mut b = a.clone();
    for (control, target) in [(1, n - 2), (n - 2, 1), (1, n - 2)] {
        b.apply_cnot(CnotGate { control, target }).unwrap();
    }
    vec![identity, a, b]
}
#[test]
fn cuda_shared_relation_preserves_mixed_nullities_wide_words_and_empty_ports() {
    for n in [6, 65, 144] {
        let c =
            SharedRelationContext::new(n, vec![1, 0], vec![1, 0], vec![vec![1], vec![2]]).unwrap();
        compare(&swaps(n), &[c], [4, 4, 3]);
        let empty = SharedRelationContext::new(n, vec![0, 1], vec![0, 1], vec![]).unwrap();
        compare(&swaps(n), &[empty], [2, 2, 1]);
    }
    let n = 144;
    let wide = SharedRelationContext::new(
        n,
        (0..65).rev().collect(),
        (0..65).collect(),
        vec![vec![u64::MAX, 1]],
    )
    .unwrap();
    compare(&swaps(n), &[wide], [70, 65, 1]);
    let coordinates = SharedRelationContext::new(
        n,
        vec![0, 64, 143],
        (0..65).rev().collect(),
        vec![vec![0, 1], vec![1, 0]],
    )
    .unwrap();
    compare(&swaps(n), &[coordinates], [3, 65, 2]);
    let n2 = SharedRelationContext::new(2, vec![0, 1], vec![0, 1], vec![]).unwrap();
    compare(
        &[MatrixState::identity(2, "arbitrary-i06").unwrap()],
        &[n2],
        [2, 2, 1],
    );
}

#[test]
fn cuda_shared_relation_preserves_context_limits_default_memory_cap_and_identity() {
    let contexts =
        [SharedRelationContext::new(6, vec![0, 1], vec![0, 1], vec![vec![1], vec![2]]).unwrap()];
    let states = swaps(6);
    for limits in [[1, 2, 2], [2, 1, 2], [2, 2, 1], [1 << 24, 2, 2]] {
        // The large channel budget exceeds the source's default64MiB estimate;
        // it must yield zero-capacity ResourceLimit before a large allocation.
        compare(&states, &contexts, limits);
    }
    let missing = [SharedRelationContext::new(3, vec![0], vec![1], vec![]).unwrap()];
    compare(&states, &missing, [2, 2, 2]);
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(&mut registry, contexts.to_vec(), 4, 4, 4).unwrap();
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
        inputs(&contexts, [3, 4, 4])
    )
    .err()
    .unwrap()
    .contains("metadata"));
    let different =
        [SharedRelationContext::new(6, vec![1, 0], vec![0, 1], vec![vec![1], vec![2]]).unwrap()];
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        6,
        0,
        inputs(&different, [4, 4, 4])
    )
    .err()
    .unwrap()
    .contains("metadata"));
    let mut relabel = policy;
    relabel.symmetry_contract = "exact_relabel".into();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &relabel,
        6,
        0,
        inputs(&contexts, [4, 4, 4])
    )
    .is_err());
}

#[test]
fn cuda_shared_relation_packet_preserves_heads_gradients_and_zero_token_behavior() {
    for (n, contexts, states) in [
        (
            6,
            vec![
                SharedRelationContext::new(6, vec![1, 0], vec![1, 0], vec![vec![1], vec![2]])
                    .unwrap(),
            ],
            swaps(6),
        ),
        (
            2,
            vec![SharedRelationContext::new(2, vec![0, 1], vec![0, 1], vec![]).unwrap()],
            vec![
                MatrixState::identity(2, "i06").unwrap(),
                MatrixState::from_packed_rows(2, "i06", vec![2, 1]).unwrap(),
            ],
        ),
    ] {
        let gpu = resident(&states);
        let mut registry = FeatureRegistry::new();
        register_shared_relation_contexts(&mut registry, contexts.clone(), 4, 4, 4).unwrap();
        let mut policy = policy();
        policy.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let plan = DeviceFeaturePlan::with_contexts(
            &registry,
            &policy,
            n,
            0,
            inputs(&contexts, [4, 4, 4]),
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
        for (x, y) in [(&a.qhat, &e.qhat), (&a.proposal_logits, &e.proposal_logits)] {
            assert!(x.masked_select(&a.action_mask).allclose(
                &y.masked_select(&e.action_mask),
                3e-5,
                3e-5,
                false
            ));
        }
        let loss = |x: &cnot103_native::model::FieldOutput| -> Tensor {
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
}
