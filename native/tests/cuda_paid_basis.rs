use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, PaidBasisPlanInputs,
};
use cnot103_native::device_paid_basis::DevicePaidBasisPlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::{
    register_paid_basis_features, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureStatus, MatrixState, PaidBasisBlockContext, PaidShearConfig,
};
use tch::{Device, Kind, Tensor};

fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Float).flatten(0, -1)).unwrap()
}
fn ints(t: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).flatten(0, -1)).unwrap()
}
fn resident(states: &[MatrixState], alphabet: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let values = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect::<Vec<_>>();
    let n = alphabet.dimension();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&values)
            .view([states.len() as i64, n as i64, n.div_ceil(64) as i64])
            .to_device(Device::Cuda(0)),
        alphabet,
        0,
    )
    .unwrap()
}
fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: (9..=14).map(|i| FeatureId::new(b'D', i).unwrap()).collect(),
        seed: 0,
        schema_id: "paid-basis-gpu-v2".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn registry(contexts: &[PaidBasisBlockContext], config: PaidShearConfig) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    register_paid_basis_features(&mut r, contexts.to_vec(), config).unwrap();
    r
}
fn inputs(
    contexts: &[PaidBasisBlockContext],
    config: PaidShearConfig,
) -> DeviceFeatureContextInputs<'_> {
    DeviceFeatureContextInputs {
        paid_basis: Some(PaidBasisPlanInputs { contexts, config }),
        ..Default::default()
    }
}
fn compare(
    states: &[MatrixState],
    alphabet: &CnotAlphabet,
    contexts: &[PaidBasisBlockContext],
    config: PaidShearConfig,
    max_bytes: Option<usize>,
) -> cnot103_native::device_paid_basis::DevicePaidBasisBatch {
    let gpu = resident(states, alphabet);
    let r = registry(contexts, config);
    let mut p = policy();
    p.max_bytes = max_bytes;
    let expected = r.build(states, &p).unwrap();
    let plan =
        DevicePaidBasisPlan::new(alphabet.dimension(), 0, contexts, config, max_bytes).unwrap();
    let actual = plan.evaluate(&gpu).unwrap();
    if ints(actual.status()).iter().all(|&v| v == 1) {
        for (sample, state) in states.iter().enumerate() {
            let n = alphabet.dimension();
            let mut stack = Vec::new();
            let mut endpoint = MatrixState::identity(n, alphabet.id()).unwrap();
            for c in contexts.iter().filter(|c| c.dimension() == n) {
                let block = c
                    .target_wires()
                    .iter()
                    .map(|&row| {
                        c.source_wires()
                            .iter()
                            .map(|&col| {
                                ((state.packed_rows()[row * n.div_ceil(64) as usize + col / 64]
                                    >> (col % 64))
                                    & 1) as u8
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<u8>>>();
                let candidate = cnot103_native::compile_paid_shear(
                    &block,
                    n as usize,
                    c.source_wires(),
                    c.target_wires(),
                    config,
                )
                .unwrap();
                for gate in candidate.certificate.gates {
                    endpoint.apply_cnot(gate).unwrap();
                    if stack.last() == Some(&gate) {
                        stack.pop();
                    } else {
                        stack.push(gate);
                    }
                }
            }
            assert_eq!(ints(actual.lengths())[sample], stack.len() as i64);
            let gates = ints(&actual.word().get(sample as i64));
            let expected = stack
                .iter()
                .flat_map(|g| [g.control as i64, g.target as i64])
                .collect::<Vec<_>>();
            assert_eq!(&gates[..expected.len()], expected);
            assert!(gates[expected.len()..].iter().all(|&v| v == 0));
            assert_eq!(
                ints(&actual.endpoint().get(sample as i64)),
                endpoint
                    .packed_rows()
                    .iter()
                    .map(|&v| v as i64)
                    .collect::<Vec<_>>()
            );
        }
    }
    for ordinal in 9..=14 {
        let id = FeatureId::new(b'D', ordinal).unwrap();
        let want = expected.feature(id);
        let value = actual.feature(ordinal).unwrap();
        assert_eq!(
            ints(actual.status()),
            want.status
                .iter()
                .map(|s| match s {
                    FeatureStatus::Computed => 1,
                    FeatureStatus::Unsupported => 3,
                    FeatureStatus::ResourceLimit => 4,
                    _ => panic!("unexpected status"),
                })
                .collect::<Vec<_>>()
        );
        if want.status.iter().all(|s| *s == FeatureStatus::Computed) {
            assert_eq!(flat(&value), want.values, "D{ordinal:02}");
            if ordinal != 14 {
                assert_eq!(
                    ints(actual.token_valid()),
                    want.token_valid
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&v| v as i64)
                        .collect::<Vec<_>>()
                );
                assert_eq!(flat(actual.incidence()), *want.incidence.as_ref().unwrap());
            }
        } else {
            assert!(flat(&value).iter().all(|&v| v == 0.0));
        }
    }
    if max_bytes.is_none() {
        let device_plan = DeviceFeaturePlan::with_contexts(
            &r,
            &p,
            alphabet.dimension(),
            0,
            inputs(contexts, config),
        )
        .unwrap();
        let packet = device_plan.build(&gpu).unwrap();
        for id in &p.enabled_ids {
            assert_eq!(
                flat(packet.feature(*id).unwrap().values()),
                flat(&actual.feature(id.ordinal()).unwrap())
            );
        }
    }
    actual
}

#[test]
fn gpu_paid_basis_preserves_all_six_features_full_word_and_noncommuting_context_order() {
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, a.id()).unwrap();
    for (control, target) in [(0, 1), (1, 2)] {
        state.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let contexts = [
        PaidBasisBlockContext::new(4, vec![0], vec![1]).unwrap(),
        PaidBasisBlockContext::new(4, vec![1], vec![2]).unwrap(),
    ];
    let forward = compare(
        &[state.clone()],
        &a,
        &contexts,
        PaidShearConfig::source_exact(),
        None,
    );
    let reverse = compare(
        &[state],
        &a,
        &[contexts[1].clone(), contexts[0].clone()],
        PaidShearConfig::source_exact(),
        None,
    );
    assert_eq!(flat(&forward.feature(14).unwrap())[(0 * 4 + 2) * 4], 1.0);
    assert_eq!(flat(&reverse.feature(14).unwrap())[(0 * 4 + 2) * 4], 0.0);
    assert_eq!(ints(forward.lengths()), [2]);
}

#[test]
fn gpu_paid_basis_keeps_empty_neighbor_boundaries_and_cascading_cancellations() {
    let a = CnotAlphabet::all_to_all_unit(8).unwrap();
    let mut state = MatrixState::identity(8, a.id()).unwrap();
    for (control, target) in [(0, 1), (2, 3), (4, 5)] {
        state.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let contexts = (0..3)
        .map(|i| PaidBasisBlockContext::new(8, vec![2 * i], vec![2 * i + 1]).unwrap())
        .chain(
            (0..3)
                .rev()
                .map(|i| PaidBasisBlockContext::new(8, vec![2 * i, 6], vec![2 * i + 1]).unwrap()),
        )
        .collect::<Vec<_>>();
    let empty = PaidBasisBlockContext::new(8, vec![6], vec![7]).unwrap();
    let adjacent = compare(
        &[state.clone()],
        &a,
        &contexts,
        PaidShearConfig::source_exact(),
        None,
    );
    let mut separated_contexts = contexts.clone();
    separated_contexts.insert(3, empty);
    let separated = compare(
        &[state],
        &a,
        &separated_contexts,
        PaidShearConfig::source_exact(),
        None,
    );
    for result in [&adjacent, &separated] {
        assert_eq!(ints(result.lengths()), [0]);
        let values = flat(&result.feature(14).unwrap());
        assert!(values.chunks_exact(4).map(|p| p[3]).sum::<f32>() > 1.0);
        for c in 0..8 {
            for t in 0..8 {
                assert_eq!(values[(c * 8 + t) * 4], f32::from(c == t));
            }
        }
    }
    assert_eq!(
        flat(&adjacent.feature(14).unwrap())
            .chunks_exact(4)
            .map(|p| p[2])
            .sum::<f32>(),
        1.0
    );
    assert_eq!(
        flat(&separated.feature(14).unwrap())
            .chunks_exact(4)
            .map(|p| p[2])
            .sum::<f32>(),
        0.0
    );
}

#[test]
fn gpu_paid_basis_preserves_width_alphabet_and_exact_workspace_budget_priority() {
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let states = [MatrixState::identity(4, a.id()).unwrap()];
    let contexts = [PaidBasisBlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap()];
    let bytes = 16 * 4 * 4 + 8 * 4 + 76;
    // These cutoffs isolate the standalone producer. Both complete packet
    // builders first check their larger estimate for the entire state batch.
    let gpu = resident(&states, &a);
    for (limit, status) in [(0, 4), (bytes - 1, 4), (bytes, 1), (bytes + 1, 1)] {
        let result = DevicePaidBasisPlan::new(
            4,
            0,
            &contexts,
            PaidShearConfig::source_exact(),
            Some(limit),
        )
        .unwrap()
        .evaluate(&gpu)
        .unwrap();
        assert_eq!(ints(result.status()), [status]);
        if status == 4 {
            for ordinal in 9..=14 {
                assert!(flat(&result.feature(ordinal).unwrap())
                    .iter()
                    .all(|&v| v == 0.0));
            }
        }
    }
    let wrong = CnotAlphabet::new(
        4,
        "paid-wrong",
        a.gates().to_vec(),
        vec![1; a.gates().len()],
    )
    .unwrap();
    let wrong_gpu = resident(&[MatrixState::identity(4, wrong.id()).unwrap()], &wrong);
    let rejected =
        DevicePaidBasisPlan::new(4, 0, &contexts, PaidShearConfig::source_exact(), Some(0))
            .unwrap()
            .evaluate(&wrong_gpu)
            .unwrap();
    assert_eq!(ints(rejected.status()), [3]);
    let other = [PaidBasisBlockContext::new(5, vec![0, 1], vec![2, 3]).unwrap()];
    let missing = DevicePaidBasisPlan::new(4, 0, &other, PaidShearConfig::source_exact(), Some(0))
        .unwrap()
        .evaluate(&gpu)
        .unwrap();
    assert_eq!(ints(missing.status()), [3]);
}

#[test]
fn gpu_paid_basis_matches_mixed_width_contexts_and_wide_physical_role_incidence() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/paid_shear_reference_v9.json")).unwrap();
    let block: Vec<Vec<u8>> = serde_json::from_value(fixture["block"].clone()).unwrap();
    let other = PaidBasisBlockContext::new(5, vec![0, 2], vec![3, 4]).unwrap();
    for (n, s, t) in [
        (10, vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8, 9]),
        (144, vec![0, 63, 64, 129], vec![1, 62, 65, 127, 128, 143]),
    ] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let mut changed = MatrixState::identity(n, a.id()).unwrap();
        for (row, values) in block.iter().enumerate() {
            for (col, &bit) in values.iter().enumerate() {
                if bit != 0 {
                    changed
                        .apply_cnot(CnotGate {
                            control: s[col] as u32,
                            target: t[row] as u32,
                        })
                        .unwrap();
                }
            }
        }
        let contexts = [
            other.clone(),
            PaidBasisBlockContext::new(n, s, t).unwrap(),
            PaidBasisBlockContext::new(n, vec![1], vec![0]).unwrap(),
        ];
        compare(
            &[MatrixState::identity(n, a.id()).unwrap(), changed],
            &a,
            &contexts,
            PaidShearConfig::source_exact(),
            None,
        );
    }
}

#[test]
fn gpu_paid_basis_binds_all_identity_fields_and_selected_ids_and_rejects_missing_metadata() {
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let contexts = [
        PaidBasisBlockContext::new(4, vec![0], vec![1]).unwrap(),
        PaidBasisBlockContext::new(5, vec![0], vec![1]).unwrap(),
    ];
    let config = PaidShearConfig::source_exact();
    let r = registry(&contexts, config);
    let p = policy();
    for changed in [
        PaidShearConfig {
            local_width: 2,
            ..config
        },
        PaidShearConfig {
            sweeps: 0,
            ..config
        },
    ] {
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, inputs(&contexts, changed))
                .err()
                .unwrap()
                .contains("metadata")
        );
    }
    for changed in [
        vec![contexts[1].clone(), contexts[0].clone()],
        vec![contexts[0].clone()],
        vec![
            contexts[0].clone(),
            PaidBasisBlockContext::new(5, vec![0], vec![2]).unwrap(),
        ],
    ] {
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, inputs(&changed, config))
                .err()
                .unwrap()
                .contains("metadata")
        );
    }
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, DeviceFeatureContextInputs::default())
            .is_err()
    );
    for (solver, symmetry) in [(false, "augmentation_only"), (true, "exact_relabel")] {
        let mut invalid = p.clone();
        invalid.solver_assisted = solver;
        invalid.symmetry_contract = symmetry.into();
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &invalid, 4, 0, inputs(&contexts, config))
                .is_err()
        );
    }
    let states = [MatrixState::identity(4, a.id()).unwrap()];
    let gpu = resident(&states, &a);
    let mut bounded = p.clone();
    bounded.max_bytes = Some(usize::MAX);
    let run = |policy| {
        DeviceFeaturePlan::with_contexts(&r, policy, 4, 0, inputs(&contexts, config))
            .unwrap()
            .build(&gpu)
            .unwrap()
    };
    let finite = run(&bounded);
    let unlimited = run(&p);
    for id in &p.enabled_ids {
        let actual = finite.feature(*id).unwrap();
        let reference = unlimited.feature(*id).unwrap();
        assert!(actual.status().equal(reference.status()));
        assert!(actual.values().equal(reference.values()));
    }
    let unlabeled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 4, 0).unwrap();
    let wrong_width = resident(
        &[MatrixState::identity(5, "all_to_all_unit_cnot:5").unwrap()],
        &CnotAlphabet::all_to_all_unit(5).unwrap(),
    );
    let standalone = DevicePaidBasisPlan::new(4, 0, &contexts, config, None).unwrap();
    assert!(standalone
        .evaluate(&unlabeled)
        .err()
        .unwrap()
        .contains("alphabet"));
    assert!(standalone.evaluate(&wrong_width).is_err());
    for ordinal in 9..=14 {
        let mut single = p.clone();
        single.enabled_ids = vec![FeatureId::new(b'D', ordinal).unwrap()];
        let actual = DeviceFeaturePlan::with_contexts(&r, &single, 4, 0, inputs(&contexts, config))
            .unwrap()
            .build(&gpu)
            .unwrap();
        let expected = r.build(&states, &single).unwrap();
        let id = single.enabled_ids[0];
        assert_eq!(
            flat(actual.feature(id).unwrap().values()),
            expected.feature(id).values
        );
    }
}

#[test]
fn gpu_paid_basis_preserves_all_model_heads_and_named_gradients() {
    use cnot103_native::model::{FieldOutput, FusionDistanceField, ModelConfig};
    use tch::nn::VarStore;
    let a = CnotAlphabet::all_to_all_unit(10).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/paid_shear_reference_v9.json")).unwrap();
    let block: Vec<Vec<u8>> = serde_json::from_value(fixture["block"].clone()).unwrap();
    let mut changed = MatrixState::identity(10, a.id()).unwrap();
    for (row, values) in block.iter().enumerate() {
        for (col, &bit) in values.iter().enumerate() {
            if bit != 0 {
                changed
                    .apply_cnot(CnotGate {
                        control: col as u32,
                        target: row as u32 + 4,
                    })
                    .unwrap();
            }
        }
    }
    let states = [MatrixState::identity(10, a.id()).unwrap(), changed];
    let gpu = resident(&states, &a);
    for contexts in [
        vec![
            PaidBasisBlockContext::new(10, vec![0, 1, 2, 3], (4..10).collect()).unwrap(),
            PaidBasisBlockContext::new(10, vec![1, 3], vec![0, 2]).unwrap(),
        ],
        vec![PaidBasisBlockContext::new(11, vec![0], vec![1]).unwrap()],
    ] {
        let config = PaidShearConfig::source_exact();
        let r = registry(&contexts, config);
        let mut p = policy();
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let actual = DeviceFeaturePlan::with_contexts(&r, &p, 10, 0, inputs(&contexts, config))
            .unwrap()
            .build(&gpu)
            .unwrap();
        let expected = r.build(&states, &p).unwrap();
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
