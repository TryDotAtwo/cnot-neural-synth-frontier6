use cnot103_native::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::device_template_response::DeviceTemplateResponsePlan;
use cnot103_native::{
    register_template_response, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureStatus, MatrixState, RationalProbability, TemplateChoice, TemplateChoiceKind,
    TemplateResponseBudgets, TemplateResponseConfig,
};
use tch::{Device, Kind, Tensor};
const IDS: [u8; 6] = [2, 3, 6, 7, 8, 9];
fn ints(t: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).flatten(0, -1)).unwrap()
}
fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Float).flatten(0, -1)).unwrap()
}
fn resident(states: &[MatrixState]) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = states[0].dimension();
    let packed = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect::<Vec<_>>();
    // This source producer is alphabet-independent. Deliberately omit a label.
    DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&packed)
            .view([states.len() as i64, n as i64, n.div_ceil(64) as i64])
            .to_device(Device::Cuda(0)),
        n,
        0,
    )
    .unwrap()
}
fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: IDS
            .into_iter()
            .map(|i| FeatureId::new(b'E', i).unwrap())
            .collect(),
        seed: 0,
        schema_id: "template-gpu-v2".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn probability(n: u64, d: u64) -> RationalProbability {
    RationalProbability::new(n, d).unwrap()
}
fn config() -> TemplateResponseConfig {
    TemplateResponseConfig::new(
        vec![
            TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
            TemplateChoice::new(TemplateChoiceKind::Swap, 0, 1),
        ],
        vec![
            vec![probability(1, 4), probability(3, 4)],
            vec![probability(1, 3), probability(1, 4)],
        ],
    )
    .unwrap()
}
fn states(n: u32) -> Vec<MatrixState> {
    let a = CnotAlphabet::all_to_all_unit(n).unwrap();
    let identity = MatrixState::identity(n, a.id()).unwrap();
    let mut one = identity.clone();
    one.apply_cnot(CnotGate {
        control: 0,
        target: n / 2,
    })
    .unwrap();
    let mut dense = identity.clone();
    for c in 0..n / 2 {
        for t in n / 2..n {
            dense
                .apply_cnot(CnotGate {
                    control: c,
                    target: t,
                })
                .unwrap();
        }
    }
    vec![identity, one, dense]
}
fn compare(states: &[MatrixState], config: &TemplateResponseConfig) {
    let n = states[0].dimension();
    let gpu = resident(states);
    let mut r = FeatureRegistry::new();
    register_template_response(&mut r, config.clone()).unwrap();
    let p = policy();
    let expected = r.build(states, &p).unwrap();
    let output = DeviceTemplateResponsePlan::new(n, 0, config, None)
        .unwrap()
        .evaluate(&gpu)
        .unwrap();
    for ordinal in IDS {
        let id = FeatureId::new(b'E', ordinal).unwrap();
        let e = expected.feature(id);
        let want = e
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Inapplicable => 2,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                _ => panic!("status"),
            })
            .collect::<Vec<_>>();
        assert_eq!(ints(output.status()), want, "E{ordinal:02}");
        let actual = output.feature(ordinal).unwrap();
        if ordinal == 6 {
            let mask = output.token_valid().to_kind(Kind::Bool);
            let values = actual.masked_select(&mask.unsqueeze(-1));
            let emask = Tensor::from_slice(e.token_valid.as_ref().unwrap())
                .to_kind(Kind::Bool)
                .view([states.len() as i64, -1]);
            let eref = Tensor::from_slice(&e.values).view([states.len() as i64, -1, 1]);
            assert_eq!(
                flat(&values),
                flat(&eref.masked_select(&emask.unsqueeze(-1)))
            );
            for (sample, &s) in want.iter().enumerate() {
                let inc = flat(&output.incidence().get(sample as i64));
                if s != 1 {
                    assert!(inc.iter().all(|&v| v == 0.0));
                    assert!(ints(&output.token_valid().get(sample as i64))
                        .iter()
                        .all(|&v| v == 0));
                } else {
                    let groups = output.token_valid().size()[1] as usize;
                    let stride = groups * n as usize * 4;
                    assert_eq!(
                        inc,
                        e.incidence.as_ref().unwrap()[sample * stride..(sample + 1) * stride]
                    );
                }
            }
        } else {
            assert_eq!(flat(&actual), e.values, "E{ordinal:02}");
        }
    }
    // Complete assignment costs from an independent direct block mutation.
    for (sample, state) in states.iter().enumerate() {
        if ints(output.status())[sample] != 1 {
            assert!(ints(&output.assignment_costs().get(sample as i64))
                .iter()
                .all(|&v| v == 0));
            continue;
        }
        let decomposition = cnot103_native::shear::partial_shear(state)
            .unwrap()
            .unwrap();
        let choices = config.choices();
        let mut expected_costs = Vec::new();
        for assignment in 0..1usize << choices.len() {
            let mut block = decomposition.block.clone();
            let mut cost = 0;
            for (i, c) in choices.iter().enumerate() {
                if (assignment >> i) & 1 == 0 {
                    continue;
                }
                match c.kind {
                    TemplateChoiceKind::Add => {
                        cost += 2;
                        for row in &mut block {
                            row[c.control] ^= row[c.target];
                        }
                    }
                    TemplateChoiceKind::Swap => {
                        cost += 6;
                        for row in &mut block {
                            row.swap(c.control, c.target);
                        }
                    }
                }
            }
            cost += block.iter().flatten().map(|&v| v as i64).sum::<i64>();
            expected_costs.push(cost);
        }
        assert_eq!(
            ints(&output.assignment_costs().get(sample as i64)),
            expected_costs
        );
    }
    let input = DeviceFeatureContextInputs {
        template_response: Some(config),
        ..Default::default()
    };
    let actual = DeviceFeaturePlan::with_contexts(&r, &p, n, 0, input)
        .unwrap()
        .build(&gpu)
        .unwrap();
    for ordinal in IDS {
        let id = FeatureId::new(b'E', ordinal).unwrap();
        assert_eq!(
            flat(actual.feature(id).unwrap().values()),
            flat(&output.feature(ordinal).unwrap())
        );
    }
}
#[test]
fn gpu_template_response_matches_full_source_fixture_and_inapplicable_rows() {
    let s = states(4);
    compare(&s, &config());
    let result = DeviceTemplateResponsePlan::new(4, 0, &config(), None)
        .unwrap()
        .evaluate(&resident(&s))
        .unwrap();
    assert_eq!(ints(&result.assignment_costs().get(1)), [1, 3, 7, 9]);
    assert_eq!(flat(&result.feature(2).unwrap().get(1)), [5.0]);
    assert_eq!(
        flat(&result.feature(3).unwrap().get(1)),
        [6.0, 19.0f32 / 6.0]
    );
    assert_eq!(ints(&result.rounding_assignments().get(1)), [0, 0, 0, 0]);
}
#[test]
fn gpu_template_response_preserves_rational_overflow_and_guard_priority() {
    let cfg = TemplateResponseConfig::new(
        (0..8)
            .map(|_| TemplateChoice::new(TemplateChoiceKind::Add, 0, 1))
            .collect(),
        vec![vec![probability(1, u64::MAX); 8]],
    )
    .unwrap();
    compare(&states(4), &cfg);
    compare(&states(2), &cfg);
    let limited = TemplateResponseConfig::with_budgets(
        cfg.choices().to_vec(),
        cfg.points().to_vec(),
        TemplateResponseBudgets {
            max_width: 1,
            ..TemplateResponseBudgets::default()
        },
    )
    .unwrap();
    compare(&states(4), &limited);
}

#[test]
fn gpu_template_response_matches_all_small_blocks_ordered_duplicates_and_max_choices() {
    let a = CnotAlphabet::all_to_all_unit(6).unwrap();
    let samples = (0..512)
        .map(|bits| {
            let mut state = MatrixState::identity(6, a.id()).unwrap();
            for r in 0..3 {
                for c in 0..3 {
                    if (bits >> (r * 3 + c)) & 1 != 0 {
                        state
                            .apply_cnot(CnotGate {
                                control: c,
                                target: r + 3,
                            })
                            .unwrap();
                    }
                }
            }
            state
        })
        .collect::<Vec<_>>();
    let choices = vec![
        TemplateChoice::new(TemplateChoiceKind::Add, 1, 2),
        TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        TemplateChoice::new(TemplateChoiceKind::Swap, 0, 2),
    ];
    let point = vec![probability(1, 3), probability(2, 5), probability(1, 7)];
    let c = TemplateResponseConfig::new(
        choices.clone(),
        vec![
            point.clone(),
            vec![probability(0, 9), probability(9, 9), probability(1, 2)],
        ],
    )
    .unwrap();
    compare(&samples, &c);
    let out = DeviceTemplateResponsePlan::new(6, 0, &c, None)
        .unwrap()
        .evaluate(&resident(&samples))
        .unwrap();
    assert!(flat(&out.feature(6).unwrap()).iter().any(|&v| v < 0.0));
    let mut reverse = choices;
    reverse.reverse();
    compare(
        &samples[..32],
        &TemplateResponseConfig::new(reverse, vec![point]).unwrap(),
    );
    let pool = [
        TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        TemplateChoice::new(TemplateChoiceKind::Swap, 1, 2),
        TemplateChoice::new(TemplateChoiceKind::Add, 2, 0),
    ];
    for k in [2, 4, 8] {
        let cfg = TemplateResponseConfig::new(
            (0..k).map(|i| pool[i % 3]).collect(),
            vec![
                (0..k)
                    .map(|i| probability((i % 3) as u64, (i % 3 + 2) as u64))
                    .collect(),
                vec![probability(1, 2); k],
            ],
        )
        .unwrap();
        compare(&states(16), &cfg);
    }
}

#[test]
fn gpu_template_response_conditional_ties_orders_and_late_overflow_are_source_exact() {
    let choices = vec![
        TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        TemplateChoice::new(TemplateChoiceKind::Add, 1, 0),
    ];
    let cfg = TemplateResponseConfig::new(choices, vec![vec![probability(1, 2); 2]]).unwrap();
    let a = CnotAlphabet::all_to_all_unit(6).unwrap();
    let mut s = MatrixState::identity(6, a.id()).unwrap();
    for c in 0..2 {
        for t in 3..6 {
            s.apply_cnot(CnotGate {
                control: c,
                target: t,
            })
            .unwrap();
        }
    }
    compare(&[s.clone()], &cfg);
    let out = DeviceTemplateResponsePlan::new(6, 0, &cfg, None)
        .unwrap()
        .evaluate(&resident(&[s]))
        .unwrap();
    assert_eq!(ints(out.assignment_costs()), [6, 5, 5, 7]);
    assert_eq!(ints(out.rounding_assignments()), [2, 1, 2]);
    let repeated = TemplateResponseConfig::new(
        vec![TemplateChoice::new(TemplateChoiceKind::Add, 0, 1); 2],
        vec![vec![probability(0, 1); 2]],
    )
    .unwrap();
    let s = states(4).pop().unwrap();
    compare(&[s.clone()], &repeated);
    let out = DeviceTemplateResponsePlan::new(4, 0, &repeated, None)
        .unwrap()
        .evaluate(&resident(&[s]))
        .unwrap();
    assert_eq!(ints(out.rounding_assignments()), [0, 0, 0]);
    let late = TemplateResponseConfig::new(
        vec![
            TemplateChoice::new(TemplateChoiceKind::Swap, 0, 1),
            TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
            TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        ],
        vec![vec![
            probability(0, 1),
            probability(1, (1u64 << 63) - 1),
            probability(1, (1u64 << 63) - 1),
        ]],
    )
    .unwrap();
    compare(&states(4), &late);
    let out = DeviceTemplateResponsePlan::new(4, 0, &late, None)
        .unwrap()
        .evaluate(&resident(&states(4)))
        .unwrap();
    assert_eq!(ints(out.status()), [2, 4, 4]);
    for ordinal in IDS {
        assert!(flat(&out.feature(ordinal).unwrap())
            .iter()
            .all(|&v| v == 0.0));
    }
    assert!(ints(out.rounding_assignments()).iter().all(|&v| v == 0));
}

#[test]
fn gpu_template_response_preserves_all_budget_boundaries_and_wide_state_words() {
    let base = config();
    let b = base.budgets();
    let width = 2;
    let assignments = 4;
    let evaluations = 35;
    let bytes = 4096;
    for budget in [
        TemplateResponseBudgets {
            max_width: width - 1,
            ..b
        },
        TemplateResponseBudgets {
            max_width: width,
            ..b
        },
        TemplateResponseBudgets {
            max_assignments: assignments - 1,
            ..b
        },
        TemplateResponseBudgets {
            max_assignments: assignments,
            ..b
        },
        TemplateResponseBudgets {
            max_evaluations: evaluations - 1,
            ..b
        },
        TemplateResponseBudgets {
            max_evaluations: evaluations,
            ..b
        },
        TemplateResponseBudgets {
            max_bytes: bytes - 1,
            ..b
        },
        TemplateResponseBudgets {
            max_bytes: bytes,
            ..b
        },
    ] {
        let c = TemplateResponseConfig::with_budgets(
            base.choices().to_vec(),
            base.points().to_vec(),
            budget,
        )
        .unwrap();
        compare(&states(4), &c);
    }
    for limit in [bytes - 1, bytes, bytes + 1] {
        let out = DeviceTemplateResponsePlan::new(4, 0, &base, Some(limit))
            .unwrap()
            .evaluate(&resident(&states(4)))
            .unwrap();
        assert_eq!(
            ints(out.status()),
            if limit < bytes {
                vec![4, 4, 4]
            } else {
                vec![2, 1, 1]
            }
        );
    }
    let huge = TemplateResponseConfig::with_budgets(
        base.choices().to_vec(),
        base.points().to_vec(),
        TemplateResponseBudgets {
            max_width: usize::MAX,
            max_assignments: usize::MAX,
            max_evaluations: usize::MAX,
            max_bytes: usize::MAX,
        },
    )
    .unwrap();
    for n in [65, 128, 144] {
        compare(&states(n), &huge);
    }
    compare(&states(65), &base);
    assert!(DeviceTemplateResponsePlan::new(1, 0, &base, None).is_err());
}

#[test]
fn gpu_template_response_binds_config_and_all_selected_ids_without_an_alphabet_requirement() {
    let base = config();
    let mut r = FeatureRegistry::new();
    register_template_response(&mut r, base.clone()).unwrap();
    let p = policy();
    fn input(c: &TemplateResponseConfig) -> DeviceFeatureContextInputs<'_> {
        DeviceFeatureContextInputs {
            template_response: Some(c),
            ..Default::default()
        }
    }
    let mut reversed = base.choices().to_vec();
    reversed.reverse();
    let mut swapped_points = base.points().to_vec();
    swapped_points.reverse();
    for changed in [
        TemplateResponseConfig::new(reversed, base.points().to_vec()).unwrap(),
        TemplateResponseConfig::new(base.choices().to_vec(), swapped_points).unwrap(),
        TemplateResponseConfig::with_budgets(
            base.choices().to_vec(),
            base.points().to_vec(),
            TemplateResponseBudgets {
                max_bytes: base.budgets().max_bytes + 1,
                ..base.budgets()
            },
        )
        .unwrap(),
    ] {
        assert!(
            DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, input(&changed))
                .err()
                .unwrap()
                .contains("metadata")
        );
    }
    assert!(
        DeviceFeaturePlan::with_contexts(&r, &p, 4, 0, DeviceFeatureContextInputs::default())
            .is_err()
    );
    let mut bad = p.clone();
    bad.symmetry_contract = "exact_relabel".into();
    assert!(DeviceFeaturePlan::with_contexts(&r, &bad, 4, 0, input(&base)).is_err());
    let gpu = resident(&states(4));
    for ordinal in IDS {
        let mut one = p.clone();
        one.max_bytes = Some(usize::MAX);
        one.enabled_ids = vec![FeatureId::new(b'E', ordinal).unwrap()];
        let actual = DeviceFeaturePlan::with_contexts(&r, &one, 4, 0, input(&base))
            .unwrap()
            .build(&gpu)
            .unwrap();
        let expected = r.build(&states(4), &one).unwrap();
        assert_eq!(
            flat(actual.feature(one.enabled_ids[0]).unwrap().values()),
            expected.feature(one.enabled_ids[0]).values
        );
    }
    let custom = states(4)
        .iter()
        .map(|s| {
            MatrixState::from_packed_rows(4, "custom-alphabet", s.packed_rows().to_vec()).unwrap()
        })
        .collect::<Vec<_>>();
    compare(&custom, &base);
    let invalid = TemplateResponseConfig::new(
        vec![TemplateChoice::new(TemplateChoiceKind::Add, usize::MAX, 0); 2],
        vec![vec![probability(1, 2); 2]],
    )
    .unwrap();
    compare(&states(4), &invalid);
    // MatrixState refuses singular inputs. Exercise a corrupted resident
    // buffer separately, rather than calling it an admitted CPU state.
    let singular = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&[1i64, 1, 4, 8])
            .view([1, 4, 1])
            .to_device(Device::Cuda(0)),
        4,
        0,
    )
    .unwrap();
    let rejected = DeviceTemplateResponsePlan::new(4, 0, &base, None)
        .unwrap()
        .evaluate(&singular)
        .unwrap();
    assert_eq!(ints(rejected.status()), [3]);
    for ordinal in IDS {
        assert!(flat(&rejected.feature(ordinal).unwrap())
            .iter()
            .all(|&v| v == 0.0));
    }
}

#[test]
fn gpu_template_response_preserves_all_model_heads_and_named_gradients() {
    use cnot103_native::model::{FieldOutput, FusionDistanceField, ModelConfig};
    use tch::nn::VarStore;
    let base = config();
    let samples = states(4);
    let gpu = resident(&samples);
    let limited = TemplateResponseConfig::with_budgets(
        base.choices().to_vec(),
        base.points().to_vec(),
        TemplateResponseBudgets {
            max_width: 1,
            ..base.budgets()
        },
    )
    .unwrap();
    for cfg in [base, limited] {
        let mut r = FeatureRegistry::new();
        register_template_response(&mut r, cfg.clone()).unwrap();
        let mut p = policy();
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let actual = DeviceFeaturePlan::with_contexts(
            &r,
            &p,
            4,
            0,
            DeviceFeatureContextInputs {
                template_response: Some(&cfg),
                ..Default::default()
            },
        )
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
