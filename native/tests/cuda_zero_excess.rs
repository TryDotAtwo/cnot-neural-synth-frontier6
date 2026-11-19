use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, GuardedPlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::device_zero_excess::DeviceZeroExcessPlan;
use cnot103_native::{
    recognize_stage37, register_guarded_zero_excess_with_budget, CnotAlphabet, CnotGate, FeatureId,
    FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState, Stage37ParentMode, Stage37Result,
};
use tch::{Device, Kind, Tensor};
fn ints(t: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).flatten(0, -1)).unwrap()
}
fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Float).flatten(0, -1)).unwrap()
}
fn resident(states: &[MatrixState], a: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let words = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect::<Vec<_>>();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&words)
            .view([
                states.len() as i64,
                a.dimension() as i64,
                a.dimension().div_ceil(64) as i64,
            ])
            .to_device(Device::Cuda(0)),
        a,
        0,
    )
    .unwrap()
}
fn inputs(max_n: u32, entries: usize) -> DeviceFeatureContextInputs<'static> {
    DeviceFeatureContextInputs {
        guarded: Some(GuardedPlanInputs {
            max_n,
            max_entries: entries,
        }),
        ..Default::default()
    }
}
fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'F', 7).unwrap()],
        seed: 0,
        schema_id: "zero-excess-gpu-v1".into(),
        solver_assisted: true,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn registry(max_n: u32, entries: usize) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    register_guarded_zero_excess_with_budget(&mut r, max_n, entries).unwrap();
    r
}
fn compare(states: &[MatrixState], a: &CnotAlphabet, max_n: u32) {
    let n = a.dimension() as usize;
    let gpu = resident(states, a);
    let p = policy();
    let r = registry(max_n, 29);
    let expected = r.build(states, &p).unwrap();
    let e = expected.feature(p.enabled_ids[0]);
    let out = DeviceZeroExcessPlan::new(n as u32, 0, max_n)
        .unwrap()
        .evaluate(&gpu)
        .unwrap();
    let packet = DeviceFeaturePlan::with_contexts(&r, &p, n as u32, 0, inputs(max_n, 29))
        .unwrap()
        .build(&gpu)
        .unwrap();
    let want = e
        .status
        .iter()
        .map(|s| match s {
            FeatureStatus::Computed => 1,
            FeatureStatus::Unsupported => 3,
            FeatureStatus::ResourceLimit => 4,
            _ => panic!("unexpected status"),
        })
        .collect::<Vec<_>>();
    assert_eq!(ints(out.status()), want);
    assert_eq!(
        ints(packet.feature(p.enabled_ids[0]).unwrap().status()),
        want
    );
    assert_eq!(flat(out.feature()), e.values);
    for (b, state) in states.iter().enumerate() {
        let headers = ints(&out.headers().get(b as i64));
        let parents = ints(&out.parents().get(b as i64));
        let changed = ints(&out.changed_rows().get(b as i64));
        let missing = ints(&out.missing_rows().get(b as i64));
        let order = ints(&out.update_order().get(b as i64));
        let gates = ints(&out.encoder_gates().get(b as i64));
        let length = ints(&out.lengths().get(b as i64))[0];
        if want[b] != 1 {
            assert_eq!(length, 0);
            assert!(headers
                .iter()
                .chain(&parents)
                .chain(&changed)
                .chain(&missing)
                .chain(&order)
                .chain(&gates)
                .all(|&v| v == 0));
            continue;
        }
        let identity = MatrixState::identity(n as u32, a.id()).unwrap();
        let changed_ref = (0..n)
            .filter(|&row| state.row_words(row as u32) != identity.row_words(row as u32))
            .map(|r| r as i64)
            .collect::<Vec<_>>();
        let mut expected_changed = changed_ref.clone();
        expected_changed.resize(n, n as i64);
        assert_eq!(changed, expected_changed);
        // Independently enumerate every full-row equality, retaining source
        // InitialRow preference even when a unit-vector FinalRow also fits.
        let mut expected_parents = vec![0i64; n * 2];
        let mut expected_missing = Vec::new();
        for row in 0..n {
            expected_parents[row * 2] = n as i64;
            expected_parents[row * 2 + 1] = 2;
            if !changed_ref.contains(&(row as i64)) {
                continue;
            }
            let mut diff = state.row_words(row as u32).unwrap().to_vec();
            diff[row / 64] ^= 1 << (row % 64);
            let parent = (0..n).find(|&c| c != row && state.row_words(c as u32).unwrap() == diff);
            let final_parent =
                (0..n).find(|&c| c != row && identity.row_words(c as u32).unwrap() == diff);
            if let Some(c) = parent {
                expected_parents[row * 2] = c as i64;
                expected_parents[row * 2 + 1] = 0;
            } else if let Some(c) = final_parent {
                expected_parents[row * 2] = c as i64;
                expected_parents[row * 2 + 1] = 1;
            } else {
                expected_missing.push(row as i64);
            }
        }
        assert_eq!(parents, expected_parents);
        let missing_count = expected_missing.len();
        expected_missing.resize(n, n as i64);
        assert_eq!(missing, expected_missing);
        match recognize_stage37(state, a).unwrap() {
            Stage37Result::Exact(exact) => {
                assert_eq!(headers, [1, changed_ref.len() as i64, 0]);
                assert_eq!(length, exact.distance as i64);
                let mut want_order = exact
                    .update_order
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>();
                want_order.resize(n, n as i64);
                assert_eq!(order, want_order);
                for parent in exact.parents {
                    let row = parent.target_row as usize;
                    assert_eq!(parents[row * 2], parent.control_row as i64);
                    assert_eq!(
                        parents[row * 2 + 1],
                        i64::from(parent.mode == Stage37ParentMode::FinalRow)
                    );
                }
                let word = exact
                    .word
                    .gates
                    .iter()
                    .rev()
                    .flat_map(|g| [g.control as i64, g.target as i64])
                    .collect::<Vec<_>>();
                assert_eq!(gates[..word.len()], word);
                assert!(gates[word.len()..].iter().all(|&v| v == 0));
                let mut replay = identity;
                for g in gates[..word.len()].chunks_exact(2) {
                    replay
                        .apply_cnot(CnotGate {
                            control: g[0] as u32,
                            target: g[1] as u32,
                        })
                        .unwrap();
                }
                assert_eq!(replay, *state);
            }
            Stage37Result::NoZeroExcess(no) => {
                assert_eq!(headers, [2, changed_ref.len() as i64, missing_count as i64]);
                assert_eq!(length, 0);
                assert_eq!(no.excluded_length, changed_ref.len() as u64);
                assert_eq!(
                    missing[..missing_count],
                    no.missing_rows
                        .iter()
                        .map(|&r| r as i64)
                        .collect::<Vec<_>>()
                );
                assert!(order.iter().all(|&v| v == n as i64));
                assert!(gates.iter().all(|&v| v == 0));
            }
            Stage37Result::Unsupported(_) => panic!("complete alphabet rejected"),
        }
    }
}
#[test]
fn gpu_zero_excess_matches_all_gl2_gl3_gl4_complete_certificates() {
    for (n, count) in [(2, 6), (3, 168), (4, 20160)] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let states = (0u32..1 << (n * n))
            .filter_map(|bits| {
                MatrixState::from_packed_rows(
                    n,
                    a.id(),
                    (0..n)
                        .map(|r| u64::from((bits >> (r * n)) & ((1 << n) - 1)))
                        .collect(),
                )
                .ok()
            })
            .collect::<Vec<_>>();
        assert_eq!(states.len(), count);
        for chunk in states.chunks(1024) {
            compare(chunk, &a, n);
        }
    }
}
#[test]
fn gpu_zero_excess_preserves_physical_words_and_parent_precedence() {
    for n in [2, 8, 65, 144] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let identity = MatrixState::identity(n, a.id()).unwrap();
        let mut chain = identity.clone();
        let mut reverse = identity.clone();
        for target in 1..n {
            chain
                .apply_cnot(CnotGate {
                    control: (target * 17 + 3) % target,
                    target,
                })
                .unwrap();
        }
        for target in (1..n).rev() {
            reverse
                .apply_cnot(CnotGate {
                    control: target,
                    target: target - 1,
                })
                .unwrap();
        }
        let mut swap = identity.clone();
        for (c, t) in [(0, n - 1), (n - 1, 0), (0, n - 1)] {
            swap.apply_cnot(CnotGate {
                control: c,
                target: t,
            })
            .unwrap();
        }
        compare(&[identity, chain, reverse, swap], &a, n);
    }
    // Both records create the same dependency 0->1: deduplicate its indegree.
    let a = CnotAlphabet::all_to_all_unit(2).unwrap();
    compare(
        &[MatrixState::from_packed_rows(2, a.id(), vec![2, 3]).unwrap()],
        &a,
        2,
    );
}
#[test]
fn gpu_zero_excess_preserves_resource_alphabet_policy_and_configuration() {
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, a.id()).unwrap();
    compare(&[state.clone()], &a, 2);
    let custom = CnotAlphabet::new(
        3,
        "zero-excess-custom",
        vec![CnotGate {
            control: 0,
            target: 1,
        }],
        vec![2],
    )
    .unwrap();
    let s = MatrixState::identity(3, custom.id()).unwrap();
    compare(&[s.clone()], &custom, 2);
    compare(&[s], &custom, 3);
    let r = registry(3, 17);
    let mut p = policy();
    for input in [inputs(4, 17), inputs(3, 18), Default::default()] {
        assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, input).is_err());
    }
    p.solver_assisted = false;
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17)).is_err());
    p.solver_assisted = true;
    for symmetry in ["exact_relabel", "augmentation_only"] {
        p.symmetry_contract = symmetry.into();
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17))
                .unwrap()
                .build(&resident(&[state.clone()], &a))
                .is_ok()
        );
    }
    p.max_bytes = Some(1);
    let gpu = resident(&[state], &a);
    let limited = DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17))
        .unwrap()
        .build(&gpu)
        .unwrap();
    let entry = limited.feature(p.enabled_ids[0]).unwrap();
    assert_eq!(ints(entry.status()), [4]);
    assert!(flat(entry.values()).iter().all(|v| *v == 0.0));
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 3, 0).unwrap();
    assert!(DeviceZeroExcessPlan::new(3, 0, 3)
        .unwrap()
        .evaluate(&unlabelled)
        .is_err());
    assert!(DeviceZeroExcessPlan::new(1, 0, 3).is_err());
    assert!(DeviceZeroExcessPlan::new(3, 0, 1).is_err());
}

#[test]
fn gpu_zero_excess_preserves_all_model_heads_and_named_gradients() {
    use cnot103_native::model::{FieldOutput, FusionDistanceField, ModelConfig};
    use tch::nn::VarStore;
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let samples = vec![
        MatrixState::identity(4, a.id()).unwrap(),
        MatrixState::from_packed_rows(4, a.id(), vec![2, 3, 6, 12]).unwrap(),
        MatrixState::from_packed_rows(4, a.id(), vec![2, 1, 4, 8]).unwrap(),
    ];
    let gpu = resident(&samples, &a);
    for max_n in [3, 4] {
        let r = registry(max_n, 100_000);
        let mut p = policy();
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let actual = DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, inputs(max_n, 100_000))
            .unwrap()
            .build(&gpu)
            .unwrap();
        let expected = r.build(&samples, &p).unwrap();
        let vars = VarStore::new(Device::Cuda(0));
        let model = FusionDistanceField::new_for_policy(
            &vars.root(),
            &r,
            &p,
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
        let x = model.forward_device(&actual).unwrap();
        let y = model.forward(&expected).unwrap();
        assert!(x.value.allclose(&y.value, 3e-5, 3e-5, false));
        for (v, w) in [(&x.qhat, &y.qhat), (&x.proposal_logits, &y.proposal_logits)] {
            assert!(v.masked_select(&x.action_mask).allclose(
                &w.masked_select(&y.action_mask),
                3e-5,
                3e-5,
                false
            ));
        }
        let loss = |o: &FieldOutput| -> Tensor {
            o.value.mean(Kind::Float)
                + o.qhat
                    .masked_select(&o.action_mask)
                    .square()
                    .mean(Kind::Float)
                    * 0.01
                + o.proposal_logits
                    .masked_select(&o.action_mask)
                    .square()
                    .mean(Kind::Float)
                    * 0.01
        };
        loss(&x).backward();
        let gradients: std::collections::BTreeMap<_, _> = vars
            .variables()
            .into_iter()
            .map(|(name, v)| {
                let grad = v.grad();
                (
                    name,
                    if grad.defined() {
                        grad.copy()
                    } else {
                        Tensor::zeros_like(&v)
                    },
                )
            })
            .collect();
        for mut v in vars.trainable_variables() {
            v.zero_grad();
        }
        loss(&y).backward();
        for (name, v) in vars.variables() {
            let grad = v.grad();
            let grad = if grad.defined() {
                grad
            } else {
                Tensor::zeros_like(&v)
            };
            assert!(
                gradients[&name].allclose(&grad, 5e-5, 5e-5, false),
                "{name}"
            );
        }
    }
}
