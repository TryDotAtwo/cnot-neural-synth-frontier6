use cnot103_native::device_feature_plan::{
    BilinearPlanInputs, DeviceFeatureContextInputs, DeviceFeaturePlan,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    register_bilinear_contexts, BilinearContext, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureStatus, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'I', 5).unwrap()],
        seed: 0,
        schema_id: "cuda-bilinear-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn inputs(contexts: &[BilinearContext], limits: [usize; 2]) -> DeviceFeatureContextInputs<'_> {
    DeviceFeatureContextInputs {
        bilinear: Some(BilinearPlanInputs {
            contexts,
            max_dimension: limits[0],
            max_matrix_bits: limits[1],
        }),
        ..Default::default()
    }
}
fn resident(states: &[MatrixState]) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = states[0].dimension();
    let data: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect();
    DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&data)
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
fn compare(states: &[MatrixState], contexts: &[BilinearContext], limits: [usize; 2]) -> Vec<f32> {
    let gpu = resident(states);
    let mut registry = FeatureRegistry::new();
    register_bilinear_contexts(&mut registry, contexts.to_vec(), limits[0], limits[1]).unwrap();
    let policy = policy();
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        states[0].dimension(),
        0,
        inputs(contexts, limits),
    )
    .unwrap();
    let actual = plan.build(&gpu).unwrap();
    let expected = registry.build(states, &policy).unwrap();
    let id = FeatureId::new(b'I', 5).unwrap();
    let a = actual.feature(id).unwrap();
    let e = expected.feature(id);
    assert_eq!(a.values().device(), Device::Cuda(0));
    assert_eq!(a.spec(), &e.spec);
    assert_eq!(flat(a.values()), e.values);
    let statuses: Vec<u8> = e
        .status
        .iter()
        .map(|s| match s {
            FeatureStatus::Computed => 1,
            FeatureStatus::Unsupported => 3,
            FeatureStatus::ResourceLimit => 4,
            s => panic!("unexpected {s:?}"),
        })
        .collect();
    assert_eq!(
        Vec::<u8>::try_from(&a.status().to_device(Device::Cpu)).unwrap(),
        statuses
    );
    e.values.clone()
}
fn one_port(n: u32) -> BilinearContext {
    BilinearContext::new(
        n,
        vec![n - 1],
        vec![0],
        2,
        vec![vec![1]],
        vec![vec![1], vec![0]],
        1,
        vec![vec![1]],
        vec![vec![0]],
    )
    .unwrap()
}
fn blocks(n: u32) -> Vec<MatrixState> {
    (0..16)
        .map(|mask| {
            let mut s = MatrixState::identity(n, "i05-custom-alphabet").unwrap();
            for row in 0..2 {
                for col in 0..2 {
                    if mask & (1 << (row * 2 + col)) != 0 {
                        s.apply_cnot(CnotGate {
                            control: col,
                            target: n - 1 - row,
                        })
                        .unwrap();
                    }
                }
            }
            s
        })
        .collect()
}
#[test]
fn cuda_bilinear_matches_all_72_pinned_stage55_goldens() {
    let rank = b"000100000001000000010001000100010000111010001110100011101110111011101000";
    let hidden = b"111100001111000011111111111111110000111100001111000011111111111111110000";
    let states = [
        MatrixState::from_packed_rows(2, "i05", vec![1, 2]).unwrap(),
        MatrixState::from_packed_rows(2, "i05", vec![1, 3]).unwrap(),
    ];
    let mut index = 0;
    for u in 1..4 {
        for v in 1..4 {
            for a0 in 0..2 {
                for d0 in 0..2 {
                    let c = BilinearContext::new(
                        2,
                        vec![1],
                        vec![0],
                        2,
                        vec![vec![u]],
                        vec![vec![v & 1], vec![(v >> 1) & 1]],
                        1,
                        vec![vec![a0]],
                        vec![vec![d0]],
                    )
                    .unwrap();
                    let values = compare(&states, &[c], [144, 100_000]);
                    for sample in 0..2 {
                        let k = sample * 36 + index;
                        assert_eq!(
                            &values[sample * 2..sample * 2 + 2],
                            &[f32::from(rank[k] - b'0'), f32::from(hidden[k] - b'0')]
                        );
                    }
                    index += 1;
                }
            }
        }
    }
    assert_eq!(index, 36);
}
#[test]
fn cuda_bilinear_block_rank_identity_matches_exhaustive_two_by_two_source_reduction() {
    let states = blocks(6);
    for v in [
        vec![vec![1], vec![2], vec![1], vec![2]],
        vec![vec![1], vec![2], vec![0], vec![2]],
        vec![vec![1], vec![2], vec![0], vec![0]],
    ] {
        for a in 0..16 {
            for d in 0..16 {
                let c = BilinearContext::new(
                    6,
                    vec![5, 4],
                    vec![0, 1],
                    4,
                    vec![vec![5], vec![10]],
                    v.clone(),
                    2,
                    vec![vec![a & 3], vec![a >> 2]],
                    vec![vec![d & 3], vec![d >> 2]],
                )
                .unwrap();
                compare(&states, &[c], [144, 100_000]);
            }
        }
    }
}
fn identity_rows(n: usize) -> Vec<Vec<u64>> {
    (0..n)
        .map(|i| {
            let mut row = vec![0; n.div_ceil(64)];
            row[i / 64] = 1u64 << (i % 64);
            row
        })
        .collect()
}
#[test]
fn cuda_bilinear_preserves_wide_and_zero_port_spaces() {
    for n in [6, 65, 144] {
        let states = blocks(n);
        let empty = BilinearContext::new(
            n,
            vec![n - 1, n - 2],
            vec![1, 0],
            0,
            vec![],
            vec![],
            0,
            vec![],
            vec![vec![], vec![]],
        )
        .unwrap();
        compare(&states, &[empty], [144, 100_000]);
        let no_u = BilinearContext::new(
            n,
            vec![n - 1, n - 2],
            vec![1, 0],
            3,
            vec![],
            vec![vec![1], vec![2], vec![0]],
            2,
            vec![],
            vec![vec![1], vec![2]],
        )
        .unwrap();
        compare(&states, &[no_u], [144, 100_000]);
        let no_v = BilinearContext::new(
            n,
            vec![n - 1, n - 2],
            vec![1, 0],
            3,
            vec![vec![1], vec![2]],
            vec![vec![], vec![], vec![]],
            0,
            vec![vec![1], vec![2]],
            vec![vec![], vec![]],
        )
        .unwrap();
        compare(&states, &[no_v], [144, 100_000]);
    }
    let mut u = identity_rows(65);
    u[0][1] ^= 1;
    let wide = BilinearContext::new(
        144,
        (65..130).rev().collect(),
        (0..65).rev().collect(),
        65,
        u,
        identity_rows(65),
        65,
        identity_rows(65),
        identity_rows(65),
    )
    .unwrap();
    compare(&blocks(144), &[wide], [144, 100_000]);
    let mut v = vec![vec![0]; 130];
    v[128][0] = 1;
    v[129][0] = 2;
    let tall = BilinearContext::new(
        6,
        vec![5, 4],
        vec![1, 0],
        130,
        vec![vec![1, 0, 1], vec![2, 0, 2]],
        v,
        2,
        vec![vec![3], vec![2]],
        vec![vec![1], vec![3]],
    )
    .unwrap();
    compare(&blocks(6), &[tall], [144, 100_000]);
}
#[test]
fn cuda_bilinear_preserves_budgets_missing_context_and_exact_identity() {
    let contexts = [one_port(65)];
    let states = blocks(65);
    for limits in [[1, 100_000], [144, 6], [2, 7]] {
        compare(&states, &contexts, limits);
    }
    compare(&states, &[one_port(2)], [144, 100_000]);
    let mut registry = FeatureRegistry::new();
    register_bilinear_contexts(&mut registry, contexts.to_vec(), 144, 100_000).unwrap();
    let policy = policy();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        65,
        0,
        DeviceFeatureContextInputs::default()
    )
    .err()
    .unwrap()
    .contains("explicit"));
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        65,
        0,
        inputs(&contexts, [143, 100_000])
    )
    .err()
    .unwrap()
    .contains("metadata"));
    let changed = [BilinearContext::new(
        65,
        vec![64],
        vec![0],
        2,
        vec![vec![1]],
        vec![vec![1], vec![0]],
        1,
        vec![vec![0]],
        vec![vec![0]],
    )
    .unwrap()];
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        65,
        0,
        inputs(&changed, [144, 100_000])
    )
    .err()
    .unwrap()
    .contains("metadata"));
    let mut exact = policy;
    exact.symmetry_contract = "exact_relabel".into();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &exact,
        65,
        0,
        inputs(&contexts, [144, 100_000])
    )
    .is_err());
}

#[test]
fn cuda_bilinear_preserves_rectangular_port_and_physical_blocks() {
    let contexts = [
        BilinearContext::new(
            6,
            vec![5, 4, 3],
            vec![1, 0],
            4,
            vec![vec![5], vec![10]],
            vec![vec![1], vec![0], vec![0], vec![1]],
            1,
            vec![vec![3], vec![2]],
            vec![vec![1], vec![0], vec![1]],
        )
        .unwrap(),
        BilinearContext::new(
            6,
            vec![5, 4],
            vec![2, 0, 1],
            4,
            vec![vec![5]],
            vec![vec![1], vec![2], vec![0], vec![1]],
            2,
            vec![vec![5]],
            vec![vec![3], vec![1]],
        )
        .unwrap(),
    ];
    for context in contexts {
        compare(&blocks(6), &[context], [144, 100_000]);
    }
}
#[test]
fn cuda_bilinear_packet_preserves_all_heads_and_named_gradients() {
    let states = blocks(6);
    let gpu = resident(&states);
    let contexts = [one_port(6)];
    let mut registry = FeatureRegistry::new();
    register_bilinear_contexts(&mut registry, contexts.to_vec(), 144, 100_000).unwrap();
    let mut policy = policy();
    policy.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        6,
        0,
        inputs(&contexts, [144, 100_000]),
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
