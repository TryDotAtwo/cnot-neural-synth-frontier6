use cnot103_native::device_endpoint_response::DeviceEndpointResponsePlan;
use cnot103_native::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    evaluate_i10, register_tiny_i10, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureStatus, I10Config, I10OracleStatus, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};
fn id() -> FeatureId {
    FeatureId::new(b'I', 10).unwrap()
}
fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id()],
        seed: 0,
        schema_id: "cuda-i10-v1".into(),
        solver_assisted: true,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn config(n: u32, degree: usize) -> I10Config {
    I10Config {
        max_n: n.max(2),
        max_degree: degree,
        max_layer_states: 5000,
        max_transitions: 100_000,
    }
}
fn inputs(config: I10Config) -> DeviceFeatureContextInputs<'static> {
    DeviceFeatureContextInputs {
        endpoint_response: Some(config),
        ..Default::default()
    }
}
fn registry(config: I10Config) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    register_tiny_i10(&mut r, config).unwrap();
    r
}
fn sample_states(a: &CnotAlphabet) -> Vec<MatrixState> {
    let n = a.dimension();
    if n <= 3 {
        return (0..1u64 << (n * n))
            .filter_map(|bits| {
                MatrixState::from_packed_rows(
                    n,
                    a.id(),
                    (0..n).map(|r| (bits >> (r * n)) & ((1 << n) - 1)).collect(),
                )
                .ok()
            })
            .collect();
    }
    let identity = MatrixState::identity(n, a.id()).unwrap();
    let mut changed = identity.clone();
    changed
        .apply_cnot(CnotGate {
            control: n - 1,
            target: 0,
        })
        .unwrap();
    let mut states = vec![identity.clone(), changed];
    if n == 4 {
        for word in [
            vec![(0, 1), (2, 3)],
            vec![(0, 1), (1, 2)],
            vec![(0, 3), (3, 0), (0, 3)],
        ] {
            let mut state = identity.clone();
            for (control, target) in word {
                state.apply_cnot(CnotGate { control, target }).unwrap();
            }
            states.push(state);
        }
    }
    states
}
fn resident(states: &[MatrixState], a: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = a.dimension();
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&x| x as i64))
        .collect();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&words)
            .view([states.len() as i64, i64::from(n), i64::from(n.div_ceil(64))])
            .to_device(Device::Cuda(0)),
        a,
        0,
    )
    .unwrap()
}
fn floats(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn words(t: &Tensor) -> Vec<u64> {
    Vec::<i64>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1]))
        .unwrap()
        .into_iter()
        .map(|x| x as u64)
        .collect()
}
fn bytes(t: &Tensor) -> Vec<u8> {
    Vec::<u8>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn wide(t: &Tensor) -> Vec<u128> {
    words(t)
        .chunks_exact(2)
        .map(|x| u128::from(x[0]) | (u128::from(x[1]) << 64))
        .collect()
}
#[derive(Debug)]
struct Record {
    status: u8,
    oracle: u8,
    degree: u64,
    leading: u128,
    terminal: u128,
    evaluations: u64,
}
fn compare(a: &CnotAlphabet, config: I10Config) -> Vec<Record> {
    let states = sample_states(a);
    let gpu = resident(&states, a);
    let r = registry(config);
    let p = policy();
    let plan = DeviceFeaturePlan::with_contexts(&r, &p, a.dimension(), 0, inputs(config)).unwrap();
    let actual = plan.build(&gpu).unwrap();
    let expected = r.build(&states, &p).unwrap();
    let x = actual.feature(id()).unwrap();
    let e = expected.feature(id());
    assert_eq!(x.spec(), &e.spec);
    assert_eq!(floats(x.values()), e.values);
    let expected_status: Vec<_> = e
        .status
        .iter()
        .map(|s| match s {
            FeatureStatus::Computed => 1,
            FeatureStatus::Unsupported => 3,
            FeatureStatus::ResourceLimit => 4,
            _ => panic!("unexpected status"),
        })
        .collect();
    assert_eq!(bytes(x.status()), expected_status);
    let raw = DeviceEndpointResponsePlan::new(a.dimension(), 0, config)
        .unwrap()
        .evaluate(&gpu)
        .unwrap();
    for t in [
        raw.values(),
        raw.status(),
        raw.oracle_status(),
        raw.leading_degrees(),
        raw.leading_counts(),
        raw.terminal_counts(),
        raw.evaluations(),
    ] {
        assert_eq!(t.device(), Device::Cuda(0));
    }
    assert_eq!(bytes(raw.status()), expected_status);
    assert_eq!(floats(raw.values()), e.values);
    let degrees = words(raw.leading_degrees());
    let leading = wide(raw.leading_counts());
    let terminal = wide(raw.terminal_counts());
    let evaluations = words(raw.evaluations());
    let status = bytes(raw.oracle_status());
    let mut records = Vec::new();
    for (b, state) in states.iter().enumerate() {
        let source = evaluate_i10(state, config);
        let oracle = match source.status {
            I10OracleStatus::Exact => 1,
            I10OracleStatus::Unsupported => 3,
            I10OracleStatus::ResourceLimit => 4,
        };
        assert_eq!(
            status[b],
            oracle,
            "n={} degree={} sample={b}",
            a.dimension(),
            config.max_degree
        );
        if oracle == 1 {
            let first = source.positive_terms.first().copied().unwrap_or((0, 0));
            assert_eq!((degrees[b], leading[b]), (first.0 as u64, first.1));
            assert_eq!(
                terminal[b],
                source
                    .positive_terms
                    .iter()
                    .find(|t| t.0 == config.max_degree)
                    .map_or(0, |t| t.1)
            );
            assert_eq!(evaluations[b], source.evaluations as u64);
        } else {
            assert_eq!(
                (degrees[b], leading[b], terminal[b], evaluations[b]),
                (0, 0, 0, 0)
            );
        }
        records.push(Record {
            status: expected_status[b],
            oracle,
            degree: degrees[b],
            leading: leading[b],
            terminal: terminal[b],
            evaluations: evaluations[b],
        });
    }
    records
}
#[test]
fn cuda_endpoint_response_matches_all_gl2_gl3_and_independent_physical_word_enumeration() {
    for n in [2, 3] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let states = sample_states(&a);
        assert_eq!(states.len(), if n == 2 { 6 } else { 168 });
        for degree in 0..=5 {
            let records = compare(&a, config(n, degree));
            let mut exact = std::collections::BTreeMap::<Vec<u64>, u128>::new();
            for mut number in 0..a.gates().len().pow(degree as u32) {
                let mut state = MatrixState::identity(n, a.id()).unwrap();
                for _ in 0..degree {
                    let gate = a.gates()[number % a.gates().len()];
                    number /= a.gates().len();
                    state.apply_cnot(gate).unwrap();
                }
                *exact.entry(state.packed_rows().to_vec()).or_default() += 1;
            }
            for (state, result) in states.iter().zip(records) {
                assert_eq!(
                    result.terminal,
                    exact.get(state.packed_rows()).copied().unwrap_or(0)
                );
            }
        }
    }
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states = sample_states(&a);
    let mut target = MatrixState::identity(3, a.id()).unwrap();
    for (control, target_wire) in [(0, 1), (1, 2)] {
        target
            .apply_cnot(CnotGate {
                control,
                target: target_wire,
            })
            .unwrap();
    }
    let index = states.iter().position(|s| s == &target).unwrap();
    for (degree, coefficient) in [(2, 1), (3, 3), (4, 23), (5, 86)] {
        let r = compare(&a, config(3, degree));
        assert_eq!(r[index].terminal, coefficient);
        assert_eq!((r[index].degree, r[index].leading), (2, 1));
        if degree == 5 {
            assert_eq!(r[index].evaluations, 1476);
        }
    }
}
#[test]
fn cuda_endpoint_response_preserves_u128_carry_overflow_and_full_horizon_guard() {
    let a = CnotAlphabet::all_to_all_unit(2).unwrap();
    for degree in [0, 1, 2, 63, 64, 65, 66, 127, 128, 129, 130, 131, usize::MAX] {
        let c = I10Config {
            max_n: 2,
            max_degree: degree,
            max_layer_states: 3,
            max_transitions: usize::MAX,
        };
        let records = compare(&a, c);
        if degree < 130 {
            assert!(records.iter().all(|r| r.oracle == 1));
            if degree >= 66 {
                assert!(records.iter().any(|r| r.terminal > (1u128 << 64)));
            }
        } else {
            assert!(records.iter().all(|r| r.status == 4 && r.oracle == 4));
        }
    }
    // The first-arrival identity answer is invalidated by a later budget failure.
    assert!(compare(
        &a,
        I10Config {
            max_n: 2,
            max_degree: 4,
            max_layer_states: 3,
            max_transitions: 17
        }
    )
    .iter()
    .all(|r| r.status == 4));
    assert!(compare(
        &a,
        I10Config {
            max_n: 2,
            max_degree: 4,
            max_layer_states: 3,
            max_transitions: 18
        }
    )
    .iter()
    .all(|r| r.status == 1));
}
#[test]
fn cuda_endpoint_response_preserves_zero_exact_and_late_layer_transition_limits() {
    let a = CnotAlphabet::all_to_all_unit(2).unwrap();
    for degree in 0..=5 {
        for layer in 0..=4 {
            for transitions in [0, 1, 2, 3, 4, 5, 6, 11, 12, 17, 18, 23, 24] {
                let records = compare(
                    &a,
                    I10Config {
                        max_n: 2,
                        max_degree: degree,
                        max_layer_states: layer,
                        max_transitions: transitions,
                    },
                );
                if degree == 0 {
                    assert!(records.iter().all(|r| r.oracle == 1));
                    assert_eq!(records.iter().filter(|r| r.status == 1).count(), 1);
                }
            }
        }
    }
    // Universe/transition allocation bounds must not invent a smaller source cap.
    compare(
        &a,
        I10Config {
            max_n: 2,
            max_degree: 6,
            max_layer_states: usize::MAX,
            max_transitions: usize::MAX,
        },
    );
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    for transitions in [1475, 1476] {
        let r = compare(
            &a,
            I10Config {
                max_n: 3,
                max_degree: 5,
                max_layer_states: 500,
                max_transitions: transitions,
            },
        );
        assert!(r
            .iter()
            .all(|x| x.oracle == if transitions == 1476 { 1 } else { 4 }));
    }
}
#[test]
fn cuda_endpoint_response_preserves_physical_keys_custom_widths_and_identity() {
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let r = compare(&a, config(4, 3));
    assert_eq!((r[2].degree, r[2].leading), (2, 2));
    let singleton = CnotAlphabet::all_to_all_unit(1).unwrap();
    assert!(DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&[1i64])
            .view([1, 1, 1])
            .to_device(Device::Cuda(0)),
        &singleton,
        0
    )
    .err()
    .unwrap()
    .contains("at least two"));
    for n in [65, 144] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let records = compare(
            &a,
            I10Config {
                max_n: n.max(2),
                max_degree: 0,
                max_layer_states: 0,
                max_transitions: 0,
            },
        );
        assert_eq!((records[0].status, records[1].status), (1, 3));
    }
    // Actual successful >64-wire layer, with every packed word part of the key.
    let a = CnotAlphabet::all_to_all_unit(65).unwrap();
    let r = compare(
        &a,
        I10Config {
            max_n: 65,
            max_degree: 1,
            max_layer_states: 4160,
            max_transitions: 4160,
        },
    );
    assert_eq!((r[1].degree, r[1].leading, r[1].terminal), (1, 1, 1));
    let a = CnotAlphabet::all_to_all_unit(144).unwrap();
    assert!(compare(
        &a,
        I10Config {
            max_n: 144,
            max_degree: 1,
            max_layer_states: 10,
            max_transitions: 20_592
        }
    )
    .iter()
    .all(|r| r.status == 4));
    assert!(compare(&a, config(3, 1)).iter().all(|r| r.status == 3));
}
#[test]
fn cuda_endpoint_response_binds_config_alphabet_solver_and_both_symmetries() {
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let c = config(3, 5);
    let r = registry(c);
    let p = policy();
    for changed in [
        I10Config { max_n: 4, ..c },
        I10Config { max_degree: 4, ..c },
        I10Config {
            max_layer_states: 5001,
            ..c
        },
        I10Config {
            max_transitions: 100001,
            ..c
        },
    ] {
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(changed))
                .err()
                .unwrap()
                .contains("metadata")
        );
    }
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, DeviceFeatureContextInputs::default())
            .is_err()
    );
    let mut raw = p.clone();
    raw.solver_assisted = false;
    assert!(DeviceFeaturePlan::with_contexts(&r, &raw, 3, 0, inputs(c)).is_err());
    let gpu = resident(&sample_states(&a), &a);
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 3, 0).unwrap();
    for symmetry in ["exact_relabel", "augmentation_only"] {
        let mut p = p.clone();
        p.symmetry_contract = symmetry.into();
        let plan = DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(c)).unwrap();
        assert!(plan.build(&unlabelled).err().unwrap().contains("alphabet"));
    }
    let wrong = CnotAlphabet::new(3, "i10-custom", a.gates().to_vec(), vec![1; 6]).unwrap();
    for c in [
        c,
        I10Config {
            max_layer_states: 0,
            max_transitions: 0,
            ..c
        },
    ] {
        assert!(compare(&wrong, c)
            .iter()
            .all(|x| x.status == 3 && x.oracle == 3));
    }
}
#[test]
fn cuda_endpoint_response_preserves_all_model_heads_and_named_gradients() {
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let states = sample_states(&a);
    let gpu = resident(&states, &a);
    for c in [
        config(4, 0),
        config(4, 3),
        I10Config {
            max_layer_states: 1,
            ..config(4, 3)
        },
    ] {
        let r = registry(c);
        let mut p = policy();
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let plan = DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, inputs(c)).unwrap();
        let actual = plan.build(&gpu).unwrap();
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
        let loss = |o: &cnot103_native::model::FieldOutput| -> Tensor {
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
        loss(&y).backward();
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
