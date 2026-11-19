use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, QuadraticPlanInputs,
};
use cnot103_native::device_quadratic::DeviceQuadraticPlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    compute_i09_response, register_i09_quadratic_response, CnotAlphabet, CnotGate, FeatureId,
    FeaturePolicy, FeatureRegistry, FeatureStatus, I09Context, I09Limits, I09Monomial, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn id() -> FeatureId {
    FeatureId::new(b'I', 9).unwrap()
}
fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id()],
        seed: 0,
        schema_id: "cuda-quadratic-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn inputs(contexts: &[I09Context], limits: I09Limits) -> DeviceFeatureContextInputs<'_> {
    DeviceFeatureContextInputs {
        quadratic: Some(QuadraticPlanInputs { contexts, limits }),
        ..Default::default()
    }
}
fn registry(contexts: &[I09Context], limits: I09Limits) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    register_i09_quadratic_response(&mut r, contexts.to_vec(), limits).unwrap();
    r
}
fn sample_states(alphabet: &CnotAlphabet) -> Vec<MatrixState> {
    let n = alphabet.dimension();
    if n == 2 {
        return (0u64..16)
            .filter_map(|bits| {
                MatrixState::from_packed_rows(2, alphabet.id(), vec![bits & 3, bits >> 2]).ok()
            })
            .collect();
    }
    let identity = MatrixState::identity(n, alphabet.id()).unwrap();
    let mut changed = identity.clone();
    changed
        .apply_cnot(CnotGate {
            control: 0,
            target: n - 1,
        })
        .unwrap();
    if n > 65 {
        changed
            .apply_cnot(CnotGate {
                control: 64,
                target: 63,
            })
            .unwrap();
    }
    vec![identity, changed]
}
fn resident(states: &[MatrixState], alphabet: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = alphabet.dimension();
    let packed: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&w| w as i64))
        .collect();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&packed)
            .view([states.len() as i64, i64::from(n), i64::from(n.div_ceil(64))])
            .to_device(Device::Cuda(0)),
        alphabet,
        0,
    )
    .unwrap()
}
fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn ints(t: &Tensor) -> Vec<i32> {
    Vec::<i32>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}
fn bytes(t: &Tensor) -> Vec<u8> {
    Vec::<u8>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn compare(
    alphabet: &CnotAlphabet,
    contexts: &[I09Context],
    limits: I09Limits,
) -> (Vec<Vec<Vec<I09Monomial>>>, Vec<u8>) {
    let states = sample_states(alphabet);
    let gpu = resident(&states, alphabet);
    let r = registry(contexts, limits);
    let p = policy();
    let plan =
        DeviceFeaturePlan::with_contexts(&r, &p, alphabet.dimension(), 0, inputs(contexts, limits))
            .unwrap();
    let actual = plan.build(&gpu).unwrap();
    let expected = r.build(&states, &p).unwrap();
    let a = actual.feature(id()).unwrap();
    let e = expected.feature(id());
    assert_eq!(a.spec(), &e.spec);
    assert_eq!(
        a.values().size(),
        e.shape.iter().map(|&v| v as i64).collect::<Vec<_>>()
    );
    assert_eq!(flat(a.values()), e.values);
    assert_eq!(flat(a.incidence().unwrap()), *e.incidence.as_ref().unwrap());
    assert_eq!(
        bytes(&a.token_valid().unwrap().to_kind(Kind::Uint8)),
        *e.token_valid.as_ref().unwrap()
    );
    let status: Vec<u8> = e
        .status
        .iter()
        .map(|s| match s {
            FeatureStatus::Computed => 1,
            FeatureStatus::Unsupported => 3,
            FeatureStatus::ResourceLimit => 4,
            s => panic!("unexpected {s:?}"),
        })
        .collect();
    assert_eq!(bytes(a.status()), status);
    let raw_plan = DeviceQuadraticPlan::new(alphabet.dimension(), 0, contexts, limits).unwrap();
    let raw = raw_plan.compile(&gpu).unwrap();
    for tensor in [
        raw.terms(),
        raw.term_counts(),
        raw.values(),
        raw.ranks(),
        raw.status(),
        raw.token_valid(),
        raw.incidence(),
    ] {
        assert_eq!(tensor.device(), Device::Cuda(0));
    }
    assert_eq!(bytes(raw.status()), status);
    assert_eq!(flat(raw.values()), e.values);
    let terms = ints(raw.terms());
    let counts = ints(raw.term_counts());
    let ranks = ints(raw.ranks());
    let g = raw.term_counts().size()[1] as usize;
    let cap = raw.terms().size()[2] as usize;
    let mut all = Vec::new();
    for (b, s) in states.iter().enumerate() {
        if status[b] != 1 {
            assert_eq!(g, 0);
            assert_eq!(&ranks[b * 2..b * 2 + 2], &[0, 0]);
            continue;
        }
        let c = contexts
            .iter()
            .find(|c| c.width() == alphabet.dimension() as usize)
            .unwrap();
        let source = compute_i09_response(s, c, limits).unwrap();
        assert_eq!(
            raw_plan.variable_count(),
            Some(source.variable_names.len() as u32)
        );
        assert_eq!(raw_plan.cut(), Some(c.cut() as u32));
        assert_eq!(
            &ranks[b * 2..b * 2 + 2],
            &[source.rank_b as i32, source.mixed_rank as i32]
        );
        let mut equations = Vec::new();
        for eq in 0..g {
            let count = counts[b * g + eq] as usize;
            assert!(count <= cap);
            let row = &terms[(b * g + eq) * cap * 3..(b * g + eq + 1) * cap * 3];
            let equation = row[..count * 3]
                .chunks_exact(3)
                .map(|t| match t[0] {
                    0 => {
                        assert_eq!(&t[1..], &[0, 0]);
                        I09Monomial::Constant
                    }
                    1 => {
                        assert_eq!(t[2], 0);
                        I09Monomial::Variable(t[1] as usize)
                    }
                    2 => {
                        assert!(t[1] < t[2]);
                        I09Monomial::Product(t[1] as usize, t[2] as usize)
                    }
                    _ => panic!("invalid monomial {t:?}"),
                })
                .collect::<Vec<_>>();
            assert_eq!(
                equation,
                source.equations[eq],
                "n={} h={} cut={} eq={eq}",
                c.width(),
                c.horizon(),
                c.cut()
            );
            assert!(row[count * 3..].iter().all(|&v| v == 0));
            equations.push(equation);
        }
        all.push(equations);
    }
    (all, status)
}
fn context(
    n: usize,
    h: usize,
    cut: usize,
    prefix: usize,
    allowed: Vec<(usize, usize)>,
) -> I09Context {
    I09Context::new(
        n,
        h,
        cut,
        (0..cut)
            .map(|i| ((prefix >> (i % usize::BITS as usize)) & 1) as u8)
            .collect(),
        allowed,
    )
    .unwrap()
}
fn all_gates(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|a| (0..n).filter(move |&b| a != b).map(move |b| (a, b)))
        .collect()
}
fn small_limits() -> I09Limits {
    I09Limits {
        max_variables: 11,
        max_equations: 19,
        max_bytes: 8192,
    }
}

#[test]
fn cuda_quadratic_matches_stage60_literals_and_independent_one_gate_word_solutions() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let c = context(2, 1, 3, 6, all_gates(2));
    let states = sample_states(&alphabet);
    let (equations, _) = compare(&alphabet, &[c.clone()], small_limits());
    let raw = DeviceQuadraticPlan::new(2, 0, &[c], small_limits())
        .unwrap()
        .compile(&resident(&states, &alphabet))
        .unwrap();
    let values = flat(raw.values());
    for (b, s) in states.iter().enumerate() {
        let mut expected = std::collections::BTreeSet::new();
        for gate in [None, Some((0, 1)), Some((1, 0))] {
            let mut rows = [1u64, 2];
            let mut assignment = 0usize;
            let readout = if let Some((control, target)) = gate {
                assignment |= 1 | (1 << (1 + control)) | (1 << (3 + target));
                let readout = rows[control];
                rows[target] ^= readout;
                readout
            } else {
                0
            };
            for j in 0..2 {
                assignment |= (((readout >> j) & 1) as usize) << (5 + j);
            }
            for i in 0..2 {
                for j in 0..2 {
                    assignment |= (((rows[i] >> j) & 1) as usize) << (7 + 2 * i + j);
                }
            }
            if s.packed_rows() == rows {
                expected.insert(assignment);
            }
        }
        let observed = (0usize..2048)
            .filter(|&assignment| {
                equations[b].iter().all(|eq| {
                    eq.iter().fold(0usize, |v, t| {
                        v ^ match *t {
                            I09Monomial::Constant => 1,
                            I09Monomial::Variable(i) => (assignment >> i) & 1,
                            I09Monomial::Product(i, j) => {
                                ((assignment >> i) & 1) & ((assignment >> j) & 1)
                            }
                        }
                    }) == 0
                })
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(observed, expected);
        let alpha: u32 = (0..16)
            .map(|eq| (values[(b * 16 + eq) * 15 + 1] as u32) << eq)
            .sum();
        let literal = match s.packed_rows() {
            [1, 2] => Some(39364),
            [1, 3] => Some(55748),
            [3, 2] => Some(47556),
            [2, 1] => Some(27076),
            _ => None,
        };
        if let Some(literal) = literal {
            assert_eq!(alpha, literal);
        }
        for eq in 0..16 {
            let v = &values[(b * 16 + eq) * 15..(b * 16 + eq + 1) * 15];
            assert_eq!(&v[2..4], &[2., 2.]);
            assert_eq!(v[4], if eq == 4 { 1. } else { 0. });
            assert_eq!(v[5], if eq == 5 { 1. } else { 0. });
            assert!(v[6..].iter().all(|&x| x == 0.));
        }
    }
}
#[test]
fn cuda_quadratic_preserves_all_small_cuts_allowed_sets_and_prefixes() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let mut cases = 0;
    for allowed_mask in 0..4 {
        let allowed: Vec<_> = all_gates(2)
            .into_iter()
            .enumerate()
            .filter_map(|(i, g)| ((allowed_mask >> i) & 1 == 1).then_some(g))
            .collect();
        for h in [1, 2] {
            let vars = h * 11;
            for cut in 0..=vars {
                let prefixes = if cut <= 4 {
                    (0..(1 << cut)).collect::<Vec<_>>()
                } else {
                    vec![0, 1, 0x555555, usize::MAX]
                };
                for prefix in prefixes {
                    let limits = if h == 1 {
                        small_limits()
                    } else {
                        I09Limits::default()
                    };
                    assert_eq!(
                        compare(
                            &alphabet,
                            &[context(2, h, cut, prefix, allowed.clone())],
                            limits
                        )
                        .1,
                        vec![1; 6]
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 648);
}
#[test]
fn cuda_quadratic_crosses_packed_variable_and_physical_boundaries_and_keeps_zero_equations() {
    for (n, h) in [(2, 7), (4, 4), (9, 1)] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let vars = h * (n as usize * n as usize + 3 * n as usize + 1);
        for cut in [0, 1, 63, 64, 65, vars] {
            for prefix in [0, 0x5555555555555555, usize::MAX] {
                assert!(compare(
                    &alphabet,
                    &[context(n as usize, h, cut, prefix, all_gates(n as usize))],
                    I09Limits::default()
                )
                .1
                .iter()
                .all(|&s| s == 1));
            }
        }
    }
    for n in [2, 65, 144] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let limits = I09Limits {
            max_variables: 1,
            max_equations: n as usize * n as usize,
            max_bytes: 256 * 1024 * 1024,
        };
        let (eqs, status) = compare(&alphabet, &[context(n as usize, 0, 0, 0, vec![])], limits);
        assert!(status.iter().all(|&s| s == 1));
        for eq in eqs.iter().flatten() {
            assert!(eq.is_empty() || *eq == vec![I09Monomial::Constant]);
        }
    }
}
#[test]
fn cuda_quadratic_preserves_limits_alphabet_order_and_full_context_identity() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let wrong = CnotAlphabet::new(2, "i09-custom", alphabet.gates().to_vec(), vec![1; 2]).unwrap();
    let c = context(2, 1, 3, 6, all_gates(2));
    for limits in [
        I09Limits {
            max_variables: 10,
            ..small_limits()
        },
        I09Limits {
            max_equations: 18,
            ..small_limits()
        },
        I09Limits {
            max_bytes: 1,
            ..small_limits()
        },
        I09Limits {
            max_variables: 1 << 24,
            ..small_limits()
        },
    ] {
        assert_eq!(compare(&alphabet, &[c.clone()], limits).1, vec![4; 6]);
        assert_eq!(compare(&wrong, &[c.clone()], limits).1, vec![3; 6]);
    }
    // Exact source working-memory threshold, found through the reference only.
    let s = sample_states(&alphabet).remove(0);
    let mut lo = 1;
    let mut hi = 8192;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if compute_i09_response(
            &s,
            &c,
            I09Limits {
                max_bytes: mid,
                ..small_limits()
            },
        )
        .is_ok()
        {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    assert_eq!(
        compare(
            &alphabet,
            &[c.clone()],
            I09Limits {
                max_bytes: lo,
                ..small_limits()
            }
        )
        .1,
        vec![1; 6]
    );
    assert_eq!(
        compare(
            &alphabet,
            &[c.clone()],
            I09Limits {
                max_bytes: lo - 1,
                ..small_limits()
            }
        )
        .1,
        vec![4; 6]
    );
    assert_eq!(
        compare(
            &CnotAlphabet::all_to_all_unit(3).unwrap(),
            &[c.clone()],
            small_limits()
        )
        .1,
        vec![3; 2]
    );
    // Large dimensions are statically limited before any state-sized equation allocation.
    for (n, h) in [(10, 1), (144, 1), (2, usize::MAX / 11)] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        assert!(compare(
            &a,
            &[context(n as usize, h, 0, 0, vec![])],
            I09Limits::default()
        )
        .1
        .iter()
        .all(|&s| s == 4));
    }
    let other = context(3, 0, 0, 0, vec![]);
    let contexts = [c.clone(), other];
    let r = registry(&contexts, small_limits());
    let p = policy();
    for altered in [
        vec![c.clone()],
        vec![context(2, 1, 3, 7, all_gates(2)), contexts[1].clone()],
        vec![context(2, 1, 3, 6, vec![(0, 1)]), contexts[1].clone()],
        vec![context(2, 2, 3, 6, all_gates(2)), contexts[1].clone()],
    ] {
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &p, 2, 0, inputs(&altered, small_limits()))
                .err()
                .unwrap()
                .contains("metadata")
        );
    }
    assert!(DeviceFeaturePlan::with_contexts(
        &r,
        &p,
        2,
        0,
        inputs(
            &contexts,
            I09Limits {
                max_bytes: 8193,
                ..small_limits()
            }
        )
    )
    .err()
    .unwrap()
    .contains("metadata"));
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &p, 2, 0, DeviceFeatureContextInputs::default())
            .is_err()
    );
    let plan =
        DeviceFeaturePlan::with_contexts(&r, &p, 2, 0, inputs(&contexts, small_limits())).unwrap();
    let gpu = resident(&sample_states(&alphabet), &alphabet);
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 2, 0).unwrap();
    assert!(plan.build(&unlabelled).err().unwrap().contains("alphabet"));
    let mut exact = p;
    exact.symmetry_contract = "exact_relabel".into();
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &exact, 2, 0, inputs(&contexts, small_limits()))
            .is_err()
    );
}
#[test]
fn cuda_quadratic_preserves_all_heads_and_named_gradients() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states = sample_states(&alphabet);
    let gpu = resident(&states, &alphabet);
    for (h, cut, bytes) in [(0, 0, 1 << 24), (2, 9, 1 << 24), (1, 3, 1)] {
        let contexts = [context(3, h, cut, usize::MAX, all_gates(3))];
        let limits = I09Limits {
            max_bytes: bytes,
            ..I09Limits::default()
        };
        let r = registry(&contexts, limits);
        let mut p = policy();
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let plan =
            DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(&contexts, limits)).unwrap();
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
