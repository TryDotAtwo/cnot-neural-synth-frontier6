//! Independent host references are confined to tests. Runtime feature assembly
//! must bind the same explicit contexts and retain every payload on CUDA.
#[path = "support/optimizer_snapshot.rs"]
mod optimizer_snapshot;
#[path = "cuda_feature_plan_contexts/record_updates.rs"]
mod record_updates;
use cnot103_native::device_feature_plan::{
    BilinearPlanInputs, BlockParityPlanInputs, DeviceFeatureContextInputs, DeviceFeaturePlan,
    FrequencyPlanInputs, MatrixBoundaryPlanInputs, PaidBasisPlanInputs, PositivePacketPlanInputs,
    QuadraticPlanInputs, SelectedResidualPlanInputs, SharedRelationPlanInputs, Stage26PlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::packet::FeatureStatus;
use cnot103_native::{
    register_basis_gradient, register_block_parities, register_general_compiler_bounds,
    register_labeled_cuts, register_matrix_boundaries, register_partial_shear_rank,
    register_rank_increment_probes, register_restricted_intersections,
    register_selected_residual_ranks, register_symmetry_alignment_h06, BlockContext, CnotAlphabet,
    CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, LabeledCut, MatrixBoundarySpec,
    MatrixState, RankIncrementProbe, RestrictedIntersection,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn policy() -> FeaturePolicy {
    let ids = [('A', 20), ('B', 12), ('C', 10), ('D', 14), ('H', 6)]
        .into_iter()
        .flat_map(|(family, count)| {
            (1..=count).map(move |i| FeatureId::new(family as u8, i).unwrap())
        })
        .chain((1..=10).map(|i| FeatureId::new(b'G', i).unwrap()))
        .chain([FeatureId::new(b'F', 1).unwrap()])
        .chain((2..=10).map(|i| FeatureId::new(b'F', i).unwrap()))
        .chain((1..=9).map(|i| FeatureId::new(b'E', i).unwrap()))
        .chain((1..=11).map(|i| FeatureId::new(b'I', i).unwrap()))
        .collect();
    FeaturePolicy {
        enabled_ids: ids,
        seed: 0,
        schema_id: "gpu-context-plan-v1".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn flat(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn assert_close(actual: &Tensor, expected: &Tensor) {
    assert!(actual.allclose(expected, 2e-5, 2e-5, false));
}

#[test]
fn contextual_plan_matches_all_102_inputs_statuses_heads_and_named_gradients() {
    check_contextual_plan_widths(&[4, 65]);
}

#[test]
fn sampled_microbatch_full_objective_with_102_inputs() {
    check_contextual_plan_widths(&[4]);
}

#[test]
fn contextual_plan_crosses_64_wire_boundary() {
    check_contextual_plan_widths(&[65]);
}

fn check_contextual_plan_widths(widths: &[u32]) {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let policy = policy();
    assert_eq!(policy.enabled_ids.len(), 102);
    assert!(!policy
        .enabled_ids
        .contains(&FeatureId::new(b'H', 7).unwrap()));
    for &n in widths {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let template = cnot103_native::TemplateResponseConfig::new(
            vec![
                cnot103_native::TemplateChoice::new(cnot103_native::TemplateChoiceKind::Add, 0, 1),
                cnot103_native::TemplateChoice::new(cnot103_native::TemplateChoiceKind::Swap, 1, 0),
            ],
            vec![vec![
                cnot103_native::RationalProbability::new(1, 3).unwrap(),
                cnot103_native::RationalProbability::new(2, 3).unwrap(),
            ]],
        )
        .unwrap();
        let endpoint = cnot103_native::I10Config {
            max_n: 4,
            max_degree: 3,
            max_layer_states: 5000,
            max_transitions: 10000,
        };
        let quadratic = vec![cnot103_native::I09Context::new(
            n as usize,
            1,
            3,
            vec![0, 1, 1],
            vec![(0, n as usize - 1)],
        )
        .unwrap()];
        let positive = vec![cnot103_native::OptionalGatePacket::new(
            n,
            vec![(0, n - 1), (1, n - 1), (0, n - 1)],
            vec![cnot103_native::LocalBooleanFactor::new(vec![0, 2], vec![]).unwrap()],
        )
        .unwrap()];
        let bilinear = vec![cnot103_native::BilinearContext::new(
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
        .unwrap()];
        let relations = vec![cnot103_native::SharedRelationContext::new(
            n,
            vec![0, 1, n - 2],
            vec![0, 1],
            vec![vec![1], vec![2]],
        )
        .unwrap()];
        let frequencies = vec![cnot103_native::FrequencyContext::new(
            n,
            vec![0],
            vec![1],
            (2..n).collect(),
            vec![vec![1]; (n - 2) as usize],
        )
        .unwrap()];
        let cuts = vec![
            LabeledCut::new(n, vec![0, 1]).unwrap(),
            LabeledCut::new(n, vec![1]).unwrap(),
        ];
        let intersections =
            vec![RestrictedIntersection::new(n, vec![0], vec![1], vec![0, 1]).unwrap()];
        let probes = vec![RankIncrementProbe::new(n, vec![0, 1], vec![0, 1], vec![1]).unwrap()];
        let gates = vec![
            CnotGate {
                control: 0,
                target: n - 1,
            },
            CnotGate {
                control: 1,
                target: n - 2,
            },
        ];
        let blocks = vec![BlockContext::new(n, vec![0, 1], vec![n - 2, n - 1]).unwrap()];
        let paid = vec![cnot103_native::PaidBasisBlockContext::new(
            n,
            vec![0, 1],
            vec![n as usize - 2, n as usize - 1],
        )
        .unwrap()];
        let boundaries = vec![MatrixBoundarySpec::new(n, vec![0, 1], vec![n - 2, n - 1])
            .unwrap()
            .with_zero_outputs(vec![n - 1])
            .unwrap()];
        let identity = MatrixState::identity(n, alphabet.id()).unwrap();
        let mut changed = identity.clone();
        for gate in [
            CnotGate {
                control: 0,
                target: n - 2,
            },
            CnotGate {
                control: 0,
                target: n - 1,
            },
            CnotGate {
                control: 1,
                target: n - 1,
            },
        ] {
            changed.apply_cnot(gate).unwrap();
        }
        let states = [identity, changed];
        let packed: Vec<i64> = states
            .iter()
            .flat_map(|s| s.packed_rows().iter().map(|v| *v as i64))
            .collect();
        let device_states = DeviceStateBatch::from_packed_tensor_for_alphabet(
            Tensor::from_slice(&packed)
                .view([2, i64::from(n), i64::from(n.div_ceil(64))])
                .to_device(Device::Cuda(0)),
            &alphabet,
            0,
        )
        .unwrap();
        for (pattern_budget, equation_budget) in [(2, 8), (8, 8), (8, 2)] {
            let mut registry = FeatureRegistry::new();
            cnot103_native::register_diffusion(&mut registry, Default::default()).unwrap();
            cnot103_native::register_template_response(&mut registry, template.clone()).unwrap();
            cnot103_native::register_paid_basis_features(
                &mut registry,
                paid.clone(),
                cnot103_native::PaidShearConfig::source_exact(),
            )
            .unwrap();
            register_symmetry_alignment_h06(&mut registry).unwrap();
            register_labeled_cuts(&mut registry, cuts.clone()).unwrap();
            register_partial_shear_rank(&mut registry).unwrap();
            register_basis_gradient(&mut registry).unwrap();
            register_general_compiler_bounds(&mut registry).unwrap();
            cnot103_native::register_stage26_features(&mut registry).unwrap();
            cnot103_native::register_guarded_pair_f02_view(&mut registry, 144, 100_000).unwrap();
            cnot103_native::register_guarded_pair(&mut registry, 144, 100_000).unwrap();
            cnot103_native::register_guarded_zero_excess_with_budget(&mut registry, 144, 100_000)
                .unwrap();
            cnot103_native::register_guarded_one_excess(&mut registry, 144, 100_000).unwrap();
            cnot103_native::register_tiny_i10(&mut registry, endpoint).unwrap();
            cnot103_native::register_i09_quadratic_response(
                &mut registry,
                quadratic.clone(),
                cnot103_native::I09Limits::default(),
            )
            .unwrap();
            cnot103_native::register_default_positive_packet_contexts(
                &mut registry,
                positive.clone(),
            )
            .unwrap();
            cnot103_native::register_default_bilinear_contexts(&mut registry, bilinear.clone())
                .unwrap();
            cnot103_native::register_default_frequency_contexts(&mut registry, frequencies.clone())
                .unwrap();
            cnot103_native::register_default_shared_relation_contexts(
                &mut registry,
                relations.clone(),
            )
            .unwrap();
            register_restricted_intersections(&mut registry, intersections.clone()).unwrap();
            register_rank_increment_probes(&mut registry, probes.clone()).unwrap();
            register_selected_residual_ranks(&mut registry, &alphabet, gates.clone()).unwrap();
            register_block_parities(&mut registry, blocks.clone(), pattern_budget, 64).unwrap();
            register_matrix_boundaries(&mut registry, boundaries.clone(), 8, equation_budget)
                .unwrap();
            let contexts = DeviceFeatureContextInputs {
                diffusion: Some(Default::default()),
                guarded: Some(cnot103_native::device_feature_plan::GuardedPlanInputs {
                    max_n: 144,
                    max_entries: 100_000,
                }),
                template_response: Some(&template),
                paid_basis: Some(PaidBasisPlanInputs {
                    contexts: &paid,
                    config: cnot103_native::PaidShearConfig::source_exact(),
                }),
                endpoint_response: Some(endpoint),
                quadratic: Some(QuadraticPlanInputs {
                    contexts: &quadratic,
                    limits: cnot103_native::I09Limits::default(),
                }),
                positive_packet: Some(PositivePacketPlanInputs {
                    contexts: &positive,
                    max_n: 144,
                    max_slots: 256,
                    max_monomials: 4096,
                    max_factor_entries: 65_536,
                }),
                bilinear: Some(BilinearPlanInputs {
                    contexts: &bilinear,
                    max_dimension: 144,
                    max_matrix_bits: 100_000,
                }),
                stage26: Some(Stage26PlanInputs::default()),
                frequency: Some(FrequencyPlanInputs {
                    contexts: &frequencies,
                    max_width: 144,
                    max_frequency_bits: 20_736,
                }),
                shared_relation: Some(SharedRelationPlanInputs {
                    contexts: &relations,
                    max_vectors: 32,
                    max_coordinates: 144,
                    max_ports: 32,
                }),
                labeled_cuts: &cuts,
                restricted_intersections: &intersections,
                rank_increment_probes: &probes,
                selected_residual: Some(SelectedResidualPlanInputs {
                    alphabet: &alphabet,
                    gates: &gates,
                }),
                selected_residual_additional_contexts: &[],
                block_parities: Some(BlockParityPlanInputs {
                    contexts: &blocks,
                    max_patterns: pattern_budget,
                    max_pairs: 64,
                }),
                matrix_boundaries: Some(MatrixBoundaryPlanInputs {
                    boundaries: &boundaries,
                    max_ports: 8,
                    max_equations: equation_budget,
                }),
            };
            let plan =
                DeviceFeaturePlan::with_contexts(&registry, &policy, n, 0, contexts).unwrap();
            let packet = plan.build(&device_states).unwrap();
            let reference = registry.build(&states, &policy).unwrap();
            for id in &policy.enabled_ids {
                let actual = packet.feature(*id).unwrap();
                let expected = reference.feature(*id);
                assert_eq!(actual.spec(), &expected.spec);
                assert_eq!(actual.values().device(), Device::Cuda(0));
                if actual.token_valid().is_none() {
                    assert_eq!(
                        flat(actual.values()),
                        expected.values,
                        "{id}, n={n}, budget={pattern_budget}/{equation_budget}"
                    );
                }
                let codes: Vec<u8> = expected
                    .status
                    .iter()
                    .map(|status| match status {
                        FeatureStatus::NotRequested => 0,
                        FeatureStatus::Computed => 1,
                        FeatureStatus::Inapplicable => 2,
                        FeatureStatus::Unsupported => 3,
                        FeatureStatus::ResourceLimit => 4,
                    })
                    .collect();
                assert_eq!(
                    Vec::<u8>::try_from(&actual.status().to_device(Device::Cpu)).unwrap(),
                    codes,
                    "{id} statuses"
                );
                if let Some(incidence) = actual.incidence() {
                    let mask = actual.token_valid().unwrap().to_kind(Kind::Bool);
                    let expected_mask = Tensor::from_slice(expected.token_valid.as_ref().unwrap())
                        .to_kind(Kind::Bool)
                        .view([2, -1]);
                    let value_mask = mask.unsqueeze(-1).expand_as(actual.values());
                    let reference_values = Tensor::from_slice(&expected.values).view([
                        2,
                        expected.shape[1] as i64,
                        expected.spec.channels as i64,
                    ]);
                    assert_eq!(
                        flat(&actual.values().masked_select(&value_mask)),
                        flat(&reference_values.masked_select(
                            &expected_mask.unsqueeze(-1).expand_as(&reference_values)
                        )),
                        "{id} valid values"
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
                    let incidence_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(incidence);
                    let reference_incidence =
                        Tensor::from_slice(expected.incidence.as_ref().unwrap()).view([
                            2,
                            expected.shape[1] as i64,
                            i64::from(n),
                            expected.spec.wire_roles.len() as i64,
                        ]);
                    assert_eq!(
                        flat(&incidence.masked_select(&incidence_mask)),
                        flat(
                            &reference_incidence.masked_select(
                                &expected_mask
                                    .unsqueeze(-1)
                                    .unsqueeze(-1)
                                    .expand_as(&reference_incidence)
                            )
                        ),
                        "{id} incidence"
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
                        "{id} token counts"
                    );
                }
            }
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
            let actual = model.forward_device(&packet).unwrap();
            let expected = model.forward(&reference).unwrap();
            if n == 4 && equation_budget == 8 {
                let root = DeviceStateBatch::from_packed_tensor_for_alphabet(
                    device_states.tensor().narrow(0, 1, 1),
                    &alphabet,
                    0,
                )
                .unwrap();
                compare_primitive_backup(
                    &model, &plan, &root, &vars, &states[1], &registry, &policy, &alphabet,
                );
            }
            assert_close(&actual.value, &expected.value);
            assert_close(
                &actual.proposal_logits.masked_select(&actual.action_mask),
                &expected
                    .proposal_logits
                    .masked_select(&expected.action_mask),
            );
            assert_close(
                &actual.qhat.masked_select(&actual.action_mask),
                &expected.qhat.masked_select(&expected.action_mask),
            );
            assert_eq!(actual.value.double_value(&[0]), 0.0);
            assert!(actual.action_mask.equal(&expected.action_mask));
            let loss = |output: &cnot103_native::model::FieldOutput| -> Tensor {
                output.value.mean(Kind::Float)
                    + output
                        .qhat
                        .masked_select(&output.action_mask)
                        .square()
                        .mean(Kind::Float)
                        * 0.01
                    + output
                        .proposal_logits
                        .masked_select(&output.action_mask)
                        .square()
                        .mean(Kind::Float)
                        * 0.01
            };
            loss(&actual).backward();
            let gradients: std::collections::BTreeMap<_, _> = vars
                .variables()
                .into_iter()
                .map(|(name, value)| {
                    let grad = value.grad();
                    (
                        name,
                        (
                            grad.defined(),
                            if grad.defined() {
                                grad.copy()
                            } else {
                                Tensor::zeros_like(&value)
                            },
                        ),
                    )
                })
                .collect();
            for mut value in vars.trainable_variables() {
                value.zero_grad();
            }
            loss(&expected).backward();
            for (name, value) in vars.variables() {
                let grad = value.grad();
                assert_eq!(
                    gradients[&name].0,
                    grad.defined(),
                    "n{n} {name} gradient presence"
                );
                let grad = if grad.defined() {
                    grad
                } else {
                    Tensor::zeros_like(&value)
                };
                assert!(
                    gradients[&name].1.allclose(&grad, 5e-5, 5e-5, false),
                    "n{n} {name} gradient"
                );
            }
            if n == 4 && (pattern_budget, equation_budget) == (2, 8) {
                let mut sampled_policy = policy.clone();
                sampled_policy.schema_id = cnot103_native::schema::MISSING_TOKEN_LAYOUT_ID.into();
                let sampled_plan =
                    DeviceFeaturePlan::with_contexts(&registry, &sampled_policy, n, 0, contexts).unwrap();
                let micro_plan =
                    DeviceFeaturePlan::with_contexts(&registry, &sampled_policy, n, 0, contexts).unwrap();
                record_updates::compare_updates(
                    plan, sampled_plan, micro_plan, &registry, &policy, &sampled_policy, &alphabet, &states,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_primitive_backup(
    model: &FusionDistanceField,
    plan: &DeviceFeaturePlan,
    root: &DeviceStateBatch,
    vars: &VarStore,
    state: &MatrixState,
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
) {
    use cnot103_native::bellman::{
        device::{DevicePrimitiveBellman, RootBranch},
        select_backup_with_macros, BellmanSelectionConfig,
    };
    let kernel = DevicePrimitiveBellman::new(alphabet, 0).unwrap();
    let reference = |strict| {
        select_backup_with_macros(
            model,
            vars,
            std::slice::from_ref(state),
            registry,
            policy,
            alphabet,
            BellmanSelectionConfig {
                macro_provider: None,
                chunk_size: Some(5),
                require_complete_model_input_coverage: strict,
            },
        )
    };
    let got = kernel
        .backup(model, plan, root, RootBranch::Nonterminal, 5, false)
        .unwrap();
    let selected = reference(false).unwrap();
    let audit = selected.branches().next().unwrap();
    assert_eq!(got.valid.int64_value(&[]), 1);
    assert_eq!(
        got.selected_index.unwrap().int64_value(&[]),
        audit.selected_index.unwrap() as i64
    );
    assert_close(
        &got.primitive_totals,
        &Tensor::from_slice(&audit.primitive_totals).to_device(vars.device()),
    );
    let expected = cnot103_native::recompute_backup(model, vars, &selected).unwrap();
    assert_close(&got.loss, &expected.loss);
    let mut params: Vec<_> = vars
        .variables()
        .into_iter()
        .filter(|(_, t)| t.requires_grad())
        .collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let inputs: Vec<_> = params.iter().map(|(_, t)| t).collect();
    let actual = Tensor::run_backward(&[&got.loss], &inputs, false, false);
    let expected = Tensor::run_backward(&[&expected.loss], &inputs, false, false);
    let slots = model.group_parameter_slots();
    for (((name, param), a), e) in params.iter().zip(actual).zip(expected) {
        let source_active = a.defined()
            && slots
                .get(&(param.data_ptr() as usize))
                .is_none_or(|&slot| got.group_activity.int64_value(&[slot]) == 1);
        assert_eq!(source_active, e.defined(), "{name}");
        if e.defined() {
            assert!(a.allclose(&e, 5e-5, 5e-5, false), "{name}");
        } else if a.defined() {
            assert_eq!(a.abs().max().double_value(&[]), 0.0, "{name}");
        }
    }
    let strict = kernel
        .backup(model, plan, root, RootBranch::Nonterminal, 5, true)
        .unwrap();
    assert_eq!(strict.valid.int64_value(&[]) == 1, reference(true).is_ok());
}

#[test]
fn contextual_plan_rejects_mismatched_provider_context_before_execution() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let cuts = vec![LabeledCut::new(4, vec![0, 1]).unwrap()];
    let wrong_cuts = vec![LabeledCut::new(4, vec![0, 2]).unwrap()];
    let mut registry = FeatureRegistry::new();
    register_labeled_cuts(&mut registry, cuts.clone()).unwrap();
    let mut policy = policy();
    policy.enabled_ids = vec![FeatureId::new(b'C', 2).unwrap()];
    let error = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        4,
        0,
        DeviceFeatureContextInputs {
            labeled_cuts: &wrong_cuts,
            ..Default::default()
        },
    )
    .err()
    .expect("a different physical cut must not carry the original producer identity");
    assert!(error.contains("metadata"), "{error}");
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        65,
        0,
        DeviceFeatureContextInputs {
            labeled_cuts: &cuts,
            ..Default::default()
        }
    )
    .is_err());
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        4,
        0,
        DeviceFeatureContextInputs::default()
    )
    .is_err());
}

#[test]
fn contextual_plan_masks_all_d08_blocks_when_only_one_exceeds_the_budget() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let blocks = vec![
        BlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap(),
        BlockContext::new(4, vec![0], vec![2, 3]).unwrap(),
    ];
    let mut registry = FeatureRegistry::new();
    register_block_parities(&mut registry, blocks.clone(), 2, 64).unwrap();
    let d08 = FeatureId::new(b'D', 8).unwrap();
    let mut selected = policy();
    selected.enabled_ids = vec![FeatureId::new(b'A', 1).unwrap(), d08];
    let inputs = DeviceFeatureContextInputs {
        block_parities: Some(BlockParityPlanInputs {
            contexts: &blocks,
            max_patterns: 2,
            max_pairs: 64,
        }),
        ..Default::default()
    };
    let plan = DeviceFeaturePlan::with_contexts(&registry, &selected, 4, 0, inputs).unwrap();
    let identity = MatrixState::identity(4, "multi-block-plan").unwrap();
    let mut changed = identity.clone();
    for (control, target) in [(0, 2), (0, 3), (1, 3)] {
        changed.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, changed];
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|v| *v as i64))
        .collect();
    let gpu_states = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&words)
            .view([2, 4, 1])
            .to_device(Device::Cuda(0)),
        4,
        0,
    )
    .unwrap();
    let primitive_context =
        cnot103_native::device_context::DeviceBlockParityContext::new(4, &blocks, 0, 2).unwrap();
    let primitive = gpu_states.block_parity_d08(&primitive_context).unwrap();
    assert_eq!(
        Vec::<u8>::try_from(&primitive.status().to_device(Device::Cpu).view([-1])).unwrap(),
        vec![1, 1, 2, 1]
    );
    let packet = plan.build(&gpu_states).unwrap();
    let actual = packet.feature(d08).unwrap();
    let reference = registry.build(&states, &selected).unwrap();
    let expected = reference.feature(d08);
    assert_eq!(
        Vec::<u8>::try_from(&actual.status().to_device(Device::Cpu)).unwrap(),
        vec![1, 4]
    );
    assert_eq!(flat(actual.values()), expected.values);
    assert_eq!(
        flat(actual.incidence().unwrap()),
        *expected.incidence.as_ref().unwrap()
    );
    assert_eq!(
        Vec::<u8>::try_from(
            &actual
                .token_valid()
                .unwrap()
                .to_kind(Kind::Uint8)
                .to_device(Device::Cpu)
                .view([-1])
        )
        .unwrap(),
        *expected.token_valid.as_ref().unwrap()
    );
    let wrong_limits = DeviceFeatureContextInputs {
        block_parities: Some(BlockParityPlanInputs {
            contexts: &blocks,
            max_patterns: 3,
            max_pairs: 64,
        }),
        ..Default::default()
    };
    assert!(
        DeviceFeaturePlan::with_contexts(&registry, &selected, 4, 0, wrong_limits)
            .err()
            .expect("resource limits are part of the producer identity")
            .contains("metadata")
    );
}
