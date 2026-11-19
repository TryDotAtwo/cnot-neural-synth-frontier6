use cnot103_native::device_feature_plan::{
    BlockParityPlanInputs, DeviceFeatureContextInputs, DeviceFeaturePlan,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::packet::FeatureStatus;
use cnot103_native::{
    register_block_parities, BlockContext, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn flat(value: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&value.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn run_case(n: u32, max_patterns: usize, max_pairs: usize, multi_context: bool, ordinals: &[u8]) {
    let source_count = if n == 144 {
        70
    } else if n == 6 {
        3
    } else {
        4
    };
    let target_count = if n == 6 { 3 } else { 6 };
    let targets: Vec<u32> = (n - target_count..n).collect();
    let mut contexts =
        vec![BlockContext::new(n, (0..source_count).collect(), targets.clone()).unwrap()];
    if multi_context {
        contexts.push(
            BlockContext::new(
                n,
                (0..source_count).step_by(2).collect(),
                targets[1..].to_vec(),
            )
            .unwrap(),
        );
    }
    let mut registry = FeatureRegistry::new();
    register_block_parities(&mut registry, contexts.clone(), max_patterns, max_pairs).unwrap();
    let selected = FeaturePolicy {
        enabled_ids: std::iter::once(FeatureId::new(b'A', 1).unwrap())
            .chain(ordinals.iter().map(|&i| FeatureId::new(b'D', i).unwrap()))
            .collect(),
        seed: 0,
        schema_id: "cuda-block-histogram-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &selected,
        n,
        0,
        DeviceFeatureContextInputs {
            block_parities: Some(BlockParityPlanInputs {
                contexts: &contexts,
                max_patterns,
                max_pairs,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let identity = MatrixState::identity(n, "block-histogram").unwrap();
    let mut repeated = identity.clone();
    for (source, target) in [
        (0, targets[0]),
        (0, targets[1]),
        (0, targets[2]),
        (1, targets[2]),
    ] {
        repeated
            .apply_cnot(CnotGate {
                control: source,
                target,
            })
            .unwrap();
    }
    let high = if n == 144 { 66 } else { source_count - 1 };
    let patterns = [
        vec![0],
        vec![0],
        vec![0, 1],
        vec![high],
        vec![0, high],
        vec![high],
    ];
    let mut nonzero = identity.clone();
    for (&target, pattern) in targets.iter().zip(patterns.iter()) {
        for &control in pattern {
            nonzero.apply_cnot(CnotGate { control, target }).unwrap();
        }
    }
    let mut mixed = nonzero.clone();
    for step in 0..(2 * n + 3) {
        let control = (17 * step + 1) % n;
        let mut target = (29 * step + 2) % n;
        if target == control {
            target = (target + 1) % n;
        }
        mixed.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, repeated, nonzero, mixed];
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|w| *w as i64))
        .collect();
    let gpu_states = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&words)
            .view([4, i64::from(n), i64::from(n.div_ceil(64))])
            .to_device(Device::Cuda(0)),
        n,
        0,
    )
    .unwrap();
    let packet = plan.build(&gpu_states).unwrap();
    let reference = registry.build(&states, &selected).unwrap();
    for id in &selected.enabled_ids {
        let actual = packet.feature(*id).unwrap();
        let expected = reference.feature(*id);
        assert_eq!(actual.spec(), &expected.spec);
        assert_eq!(actual.values().device(), Device::Cuda(0));
        let codes: Vec<u8> = expected
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::ResourceLimit => 4,
                other => panic!("unexpected block status: {other:?}"),
            })
            .collect();
        assert_eq!(
            Vec::<u8>::try_from(&actual.status().to_device(Device::Cpu)).unwrap(),
            codes,
            "{id} status n={n}, budgets={max_patterns}/{max_pairs}"
        );
        if let Some(mask) = actual.token_valid() {
            let mask = mask.to_kind(Kind::Bool);
            let expected_mask = Tensor::from_slice(expected.token_valid.as_ref().unwrap())
                .to_kind(Kind::Bool)
                .view([4, -1]);
            let value_mask = mask.unsqueeze(-1).expand_as(actual.values());
            let expected_values = Tensor::from_slice(&expected.values).view([
                4,
                expected.shape[1] as i64,
                expected.spec.channels as i64,
            ]);
            assert_eq!(
                flat(&actual.values().masked_select(&value_mask)),
                flat(
                    &expected_values
                        .masked_select(&expected_mask.unsqueeze(-1).expand_as(&expected_values))
                ),
                "{id} canonical values, n={n}"
            );
            assert!(
                actual
                    .values()
                    .masked_select(&value_mask.logical_not())
                    .eq(0)
                    .all()
                    .int64_value(&[])
                    != 0
            );
            let incidence = actual.incidence().unwrap();
            let incidence_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(incidence);
            let expected_incidence = Tensor::from_slice(expected.incidence.as_ref().unwrap())
                .view([4, expected.shape[1] as i64, i64::from(n), 3]);
            assert_eq!(
                flat(&incidence.masked_select(&incidence_mask)),
                flat(
                    &expected_incidence.masked_select(
                        &expected_mask
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                            .expand_as(&expected_incidence)
                    )
                ),
                "{id} physical roles n={n}"
            );
            assert!(
                incidence
                    .masked_select(&incidence_mask.logical_not())
                    .eq(0)
                    .all()
                    .int64_value(&[])
                    != 0
            );
            assert_eq!(
                Vec::<i64>::try_from(
                    &mask
                        .sum_dim_intlist([1i64].as_slice(), false, Kind::Int64)
                        .to_device(Device::Cpu)
                )
                .unwrap(),
                Vec::<i64>::try_from(&expected_mask.sum_dim_intlist(
                    [1i64].as_slice(),
                    false,
                    Kind::Int64
                ))
                .unwrap(),
                "{id} per-sample counts"
            );
        } else {
            assert_eq!(flat(actual.values()), expected.values);
        }
    }
    // Independent literal case: rows [1,1,3] imply zero:0, one:2, three:1.
    if !multi_context
        && n == 6
        && max_patterns >= 3
        && max_pairs >= 1
        && ordinals.contains(&1)
        && ordinals.contains(&7)
    {
        let d01 = packet.feature(FeatureId::new(b'D', 1).unwrap()).unwrap();
        assert_eq!(
            flat(
                &d01.values().select(0, 1).masked_select(
                    &d01.token_valid()
                        .unwrap()
                        .select(0, 1)
                        .to_kind(Kind::Bool)
                        .unsqueeze(-1)
                )
            ),
            vec![0.0, 2.0, 1.0]
        );
        assert_eq!(
            flat(
                &packet
                    .feature(FeatureId::new(b'D', 7).unwrap())
                    .unwrap()
                    .values()
                    .select(0, 1)
            ),
            vec![2.0, 2.0, 3.0, 1.0, 2.0]
        );
    }
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &registry,
        &selected,
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
    let actual = model.forward_device(&packet).unwrap();
    let expected = model.forward(&reference).unwrap();
    assert!(actual.value.allclose(&expected.value, 3e-5, 3e-5, false));
    assert!(actual.qhat.masked_select(&actual.action_mask).allclose(
        &expected.qhat.masked_select(&expected.action_mask),
        3e-5,
        3e-5,
        false
    ));
    assert!(actual
        .proposal_logits
        .masked_select(&actual.action_mask)
        .allclose(
            &expected
                .proposal_logits
                .masked_select(&expected.action_mask),
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
    loss(&actual).backward();
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
    loss(&expected).backward();
    for (name, v) in vars.variables() {
        let g = v.grad();
        let expected = if g.defined() {
            g
        } else {
            Tensor::zeros_like(&v)
        };
        assert!(
            gradients[&name].allclose(&expected, 5e-5, 5e-5, false),
            "block gradient {name}, n={n}"
        );
    }
}

#[test]
fn cuda_block_histograms_match_canonical_patterns_roles_and_gradients() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    run_case(6, 16, 16, false, &[1, 2, 4, 6, 7]);
    for n in [6, 65, 144] {
        run_case(n, 16, 16, true, &[1, 2, 4, 6, 7]);
    }
}

#[test]
fn cuda_block_histograms_preserve_distinct_pattern_and_pair_only_limits() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    for (patterns, pairs) in [(2, 16), (16, 0), (16, 1)] {
        run_case(144, patterns, pairs, true, &[1, 2, 4, 6, 7]);
    }
}

#[test]
fn cuda_block_histogram_individual_consumers_skip_unused_outputs() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    run_case(65, 16, 16, true, &[2]);
    run_case(65, 16, 16, true, &[6]);
    run_case(65, 16, 0, true, &[6]);
    run_case(144, 16, 16, true, &[7]);
}
