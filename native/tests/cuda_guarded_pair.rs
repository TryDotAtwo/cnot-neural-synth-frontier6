use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, GuardedPlanInputs,
};
use cnot103_native::device_guarded_pair::DeviceGuardedPairPlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::{
    recognize_stage35, register_guarded_pair, register_guarded_pair_f02_view, CnotAlphabet,
    CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState, Stage35Branch,
    Stage35LowerBoundKind, Stage35Result,
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
    let data = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect::<Vec<_>>();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&data)
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
fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: [2, 6].map(|i| FeatureId::new(b'F', i).unwrap()).to_vec(),
        seed: 0,
        schema_id: "guarded-pair-cuda-v1".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn registry(max_n: u32, entries: usize) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    register_guarded_pair_f02_view(&mut r, max_n, entries).unwrap();
    register_guarded_pair(&mut r, max_n, entries).unwrap();
    r
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
fn fixture(n: u32, active: [u32; 2], frame: [u8; 2], columns: &[(u32, u8)]) -> MatrixState {
    let w = n.div_ceil(64) as usize;
    let mut data = MatrixState::identity(n, format!("all_to_all_unit_cnot:{n}"))
        .unwrap()
        .packed_rows()
        .to_vec();
    for r in 0..2 {
        data[active[r] as usize * w..(active[r] as usize + 1) * w].fill(0);
        for c in 0..2 {
            if frame[r] & (1 << c) != 0 {
                data[active[r] as usize * w + active[c] as usize / 64] |= 1 << (active[c] % 64);
            }
        }
        for &(c, v) in columns {
            if (frame[r] & v).count_ones() % 2 != 0 {
                data[active[r] as usize * w + c as usize / 64] |= 1 << (c % 64);
            }
        }
    }
    MatrixState::from_packed_rows(n, format!("all_to_all_unit_cnot:{n}"), data).unwrap()
}
fn compare(states: &[MatrixState], a: &CnotAlphabet, max_n: u32) {
    let n = a.dimension();
    let gpu = resident(states, a);
    let r = registry(max_n, 37);
    let p = policy();
    let cpu = r.build(states, &p).unwrap();
    let output = DeviceGuardedPairPlan::new(n, 0, max_n)
        .unwrap()
        .evaluate(&gpu)
        .unwrap();
    let packet = DeviceFeaturePlan::with_contexts(&r, &p, n, 0, inputs(max_n, 37))
        .unwrap()
        .build(&gpu)
        .unwrap();
    let statuses = ints(output.status());
    let lengths = ints(output.lengths());
    for ordinal in [2, 6] {
        let id = FeatureId::new(b'F', ordinal).unwrap();
        let e = cpu.feature(id);
        let want = e
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                _ => panic!("status"),
            })
            .collect::<Vec<_>>();
        assert_eq!(statuses, want);
        assert_eq!(ints(packet.feature(id).unwrap().status()), want);
        let v = if ordinal == 2 {
            output.f02()
        } else {
            output.f06()
        };
        if ordinal == 2 || want.contains(&1) {
            assert_eq!(flat(v), e.values);
        } else {
            assert!(flat(v).iter().all(|&v| v == 0.0));
        }
    }
    for (b, state) in states.iter().enumerate() {
        let c = ints(&output.certificate().get(b as i64));
        let gates = ints(&output.encoder_gates().get(b as i64));
        if statuses[b] != 1 {
            assert_eq!(lengths[b], 0);
            assert!(c.iter().chain(&gates).all(|&v| v == 0));
            assert_eq!(ints(&output.token_valid().get(b as i64)), [0]);
            assert!(flat(&output.incidence().get(b as i64))
                .iter()
                .all(|&v| v == 0.0));
            continue;
        }
        let Stage35Result::Exact(e) = recognize_stage35(state, a).unwrap() else {
            panic!("expected exact")
        };
        let absent = i64::from(n);
        assert_eq!(
            c,
            vec![
                e.changed_rows.len() as i64,
                e.active_rows[0] as i64,
                e.active_rows[1] as i64,
                e.frame_rows[0] as i64,
                e.frame_rows[1] as i64,
                e.nonzero_external_columns as i64,
                e.shared_external_columns as i64,
                i64::from(e.branch == Stage35Branch::NonemptyFrameMotion),
                e.connectivity as i64,
                e.zero_diagonal as i64,
                e.lower_bound as i64,
                match e.lower_bound_kind {
                    Stage35LowerBoundKind::ConnectivityPlusDiagonalZeros => 0,
                    Stage35LowerBoundKind::TreeObstructionTwoSharedSources => 1,
                    Stage35LowerBoundKind::Connectivity => 2,
                },
                e.shared_sources.first().map_or(absent, |&x| i64::from(x)),
                e.shared_sources.get(1).map_or(absent, |&x| i64::from(x)),
                e.changed_rows.first().map_or(absent, |&x| i64::from(x)),
                e.changed_rows.get(1).map_or(absent, |&x| i64::from(x))
            ]
        );
        assert_eq!(lengths[b], e.distance as i64);
        let want = e
            .encoder_gates
            .iter()
            .flat_map(|g| [i64::from(g.control), i64::from(g.target)])
            .collect::<Vec<_>>();
        assert_eq!(gates[..want.len()], want);
        assert!(gates[want.len()..].iter().all(|&v| v == 0));
        let mut replay = MatrixState::identity(n, a.id()).unwrap();
        for g in gates[..want.len()].chunks_exact(2) {
            replay
                .apply_cnot(CnotGate {
                    control: g[0] as u32,
                    target: g[1] as u32,
                })
                .unwrap();
        }
        assert_eq!(replay, *state);
        assert_eq!(ints(&output.token_valid().get(b as i64)), [1]);
        let ef = cpu.feature(FeatureId::new(b'F', 6).unwrap());
        let stride = n as usize * 3;
        assert_eq!(
            flat(&output.incidence().get(b as i64)),
            ef.incidence.as_ref().unwrap()[b * stride..(b + 1) * stride]
        );
    }
}
#[test]
fn cuda_guarded_pair_preserves_all_gl3_and_all_small_frames_words_and_certificates() {
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states = (0..512)
        .filter_map(|bits| {
            MatrixState::from_packed_rows(
                3,
                a.id(),
                (0..3).map(|r| ((bits >> (r * 3)) & 7) as u64).collect(),
            )
            .ok()
        })
        .collect::<Vec<_>>();
    assert_eq!(states.len(), 168);
    compare(&states, &a, 3);
    let a = CnotAlphabet::all_to_all_unit(6).unwrap();
    let mut states = Vec::new();
    for frame in [[1, 2], [1, 3], [3, 2], [2, 3], [3, 1], [2, 1]] {
        for mask in 0..256 {
            states.push(fixture(
                6,
                [0, 1],
                frame,
                &(0..4)
                    .map(|i| (i + 2, ((mask >> (i * 2)) & 3) as u8))
                    .collect::<Vec<_>>(),
            ));
        }
    }
    compare(&states, &a, 6);
}
#[test]
fn cuda_guarded_pair_preserves_physical_order_and_cross_word_sources() {
    for n in [2, 3, 65, 144] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let mut states = vec![MatrixState::identity(n, a.id()).unwrap()];
        for frame in [[1, 2], [1, 3], [3, 2], [2, 3], [3, 1], [2, 1]] {
            let active = [n - 1, 0];
            let ext = (1..n - 1).map(|c| (c, (c % 4) as u8)).collect::<Vec<_>>();
            states.push(fixture(n, active, frame, &ext));
        }
        compare(&states, &a, n);
    }
}
#[test]
fn cuda_guarded_pair_preserves_guards_policy_and_metadata() {
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, a.id()).unwrap();
    compare(&[state.clone()], &a, 2);
    let custom = CnotAlphabet::new(
        3,
        "guarded-custom",
        vec![CnotGate {
            control: 0,
            target: 1,
        }],
        vec![2],
    )
    .unwrap();
    let custom_state = MatrixState::identity(3, custom.id()).unwrap();
    compare(&[custom_state.clone()], &custom, 2); // width takes precedence over alphabet
    compare(&[custom_state], &custom, 3);
    let r = registry(3, 17);
    let mut p = policy();
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 18)).is_err());
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(4, 17)).is_err());
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, Default::default()).is_err());
    p.solver_assisted = false;
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17)).is_err());
    p.solver_assisted = true;
    p.symmetry_contract = "exact_relabel".into();
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17)).is_err());
    p.enabled_ids = vec![FeatureId::new(b'F', 2).unwrap()];
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17))
            .unwrap()
            .build(&resident(&[state.clone()], &a))
            .is_ok()
    );
    p.enabled_ids = vec![FeatureId::new(b'F', 6).unwrap()];
    p.symmetry_contract = "augmentation_only".into();
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17))
            .unwrap()
            .build(&resident(&[state], &a))
            .is_ok()
    );
    assert!(DeviceGuardedPairPlan::new(1, 0, 3).is_err());
    assert!(DeviceGuardedPairPlan::new(3, 0, 1).is_err());
}

#[test]
fn cuda_guarded_pair_preserves_all_model_heads_and_named_gradients() {
    use cnot103_native::model::{FieldOutput, FusionDistanceField, ModelConfig};
    use tch::nn::VarStore;
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let samples = vec![
        MatrixState::identity(3, a.id()).unwrap(),
        MatrixState::from_packed_rows(3, a.id(), vec![2, 1, 4]).unwrap(),
        MatrixState::from_packed_rows(3, a.id(), vec![2, 4, 1]).unwrap(),
    ];
    let gpu = resident(&samples, &a);
    for max_n in [2, 3] {
        let r = registry(max_n, 100_000);
        let mut p = policy();
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let actual = DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(max_n, 100_000))
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
