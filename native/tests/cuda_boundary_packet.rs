use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, MatrixBoundaryPlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::matrix_boundary::RationalWeight;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::packet::FeatureStatus;
use cnot103_native::{
    register_matrix_boundaries, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    MatrixBoundarySpec, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn flat(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&tensor.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn run_case(n: u32, wide: bool, zero_outputs: bool, weighted: bool, limited: bool) {
    let input_count = if wide { n } else { 3.min(n) };
    let output_count = if wide { n - 1 } else { 3.min(n) };
    let inputs = (0..input_count).collect::<Vec<_>>();
    let outputs = (n - output_count..n).collect::<Vec<_>>();
    let mut boundary = MatrixBoundarySpec::new(n, inputs.clone(), outputs.clone())
        .unwrap()
        .with_protected_inputs(vec![0, input_count - 1])
        .unwrap();
    if zero_outputs {
        boundary = boundary
            .with_zero_outputs(vec![outputs[0], *outputs.last().unwrap()])
            .unwrap();
    }
    if weighted {
        boundary = boundary
            .with_weights(vec![
                (
                    RationalWeight::new(2, 3).unwrap(),
                    RationalWeight::new(5, 7).unwrap()
                );
                outputs.len()
            ])
            .unwrap();
    }
    let boundaries = vec![boundary];
    let max_ports = if limited { 1 } else { 4 * n as usize };
    let max_equations = 4 * n as usize;
    let mut registry = FeatureRegistry::new();
    register_matrix_boundaries(&mut registry, boundaries.clone(), max_ports, max_equations)
        .unwrap();
    let selected = FeaturePolicy {
        enabled_ids: [
            (b'A', 1),
            (b'C', 8),
            (b'I', 1),
            (b'I', 2),
            (b'I', 3),
            (b'I', 4),
            (b'I', 11),
        ]
        .map(|(f, i)| FeatureId::new(f, i).unwrap())
        .to_vec(),
        seed: 0,
        schema_id: "gpu-boundary-packet-v1".into(),
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
            matrix_boundaries: Some(MatrixBoundaryPlanInputs {
                boundaries: &boundaries,
                max_ports,
                max_equations,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let states = (0..4)
        .map(|seed| {
            let mut state = MatrixState::identity(n, "boundary-packet").unwrap();
            if seed != 0 {
                for step in 0..(2 * n + seed) {
                    let control = (17 * step + seed) % n;
                    let mut target = (23 * step + seed + 1) % n;
                    if target == control {
                        target = (target + 1) % n;
                    }
                    state.apply_cnot(CnotGate { control, target }).unwrap();
                }
            }
            state
        })
        .collect::<Vec<_>>();
    let words = states
        .iter()
        .flat_map(|state| state.packed_rows().iter().map(|word| *word as i64))
        .collect::<Vec<_>>();
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
        let codes = expected
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1u8,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                other => panic!("unexpected boundary status {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            Vec::<u8>::try_from(&actual.status().to_device(Device::Cpu)).unwrap(),
            codes,
            "{id} status"
        );
        if let Some(mask) = actual.token_valid() {
            // GPU uses fixed capacity; reference uses the batch's maximum rank.
            // Only valid tokens are semantic data; every unused GPU slot must be zero.
            let mask = mask.to_kind(Kind::Bool);
            let value_mask = mask
                .unsqueeze(-1)
                .expand(actual.values().size().as_slice(), true);
            let expected_mask =
                Tensor::from_slice(expected.token_valid.as_ref().unwrap()).to_kind(Kind::Bool);
            let expected_values =
                Tensor::from_slice(&expected.values).view([-1, expected.spec.channels as i64]);
            assert_eq!(
                flat(&actual.values().masked_select(&value_mask)),
                flat(
                    &expected_values
                        .masked_select(&expected_mask.unsqueeze(-1).expand_as(&expected_values))
                ),
                "{id} values"
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
            let incidence_mask = mask.view([4, -1, 1, 1]).expand_as(incidence);
            let expected_incidence = Tensor::from_slice(expected.incidence.as_ref().unwrap())
                .view([-1, i64::from(n), expected.spec.wire_roles.len() as i64]);
            assert_eq!(
                flat(&incidence.masked_select(&incidence_mask)),
                flat(
                    &expected_incidence.masked_select(
                        &expected_mask
                            .view([-1, 1, 1])
                            .expand_as(&expected_incidence)
                    )
                ),
                "{id} canonical incidence"
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
                mask.sum(Kind::Int64).int64_value(&[]),
                expected_mask.sum(Kind::Int64).int64_value(&[])
            );
        } else {
            assert_eq!(flat(actual.values()), expected.values, "{id} values");
        }
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
    assert_eq!(actual.value.double_value(&[0]), 0.0);
    let loss = |output: &cnot103_native::model::FieldOutput| {
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
    let device_gradients = vars
        .variables()
        .into_iter()
        .map(|(name, variable)| {
            let grad = variable.grad();
            (
                name,
                if grad.defined() {
                    grad.copy()
                } else {
                    Tensor::zeros_like(&variable)
                },
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for mut variable in vars.trainable_variables() {
        variable.zero_grad();
    }
    loss(&expected).backward();
    for (name, variable) in vars.variables() {
        let grad = variable.grad();
        let expected = if grad.defined() {
            grad
        } else {
            Tensor::zeros_like(&variable)
        };
        assert!(
            device_gradients[&name].allclose(&expected, 5e-5, 5e-5, false),
            "boundary gradient: {name}, n={n}"
        );
    }
}

#[test]
fn cuda_boundary_canonical_relations_match_reference_across_packed_words() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    for (n, wide) in [(4, false), (65, false), (65, true), (144, true)] {
        run_case(n, wide, true, false, false);
        run_case(n, wide, false, true, false);
    }
}

#[test]
fn cuda_boundary_status_and_zero_padding_preserve_reference_limits() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    run_case(4, false, false, false, false);
    run_case(4, false, true, true, false);
    run_case(4, false, true, false, true);
}

#[test]
fn wide_boundary_packets_match_every_scalar_relation_and_protected_rank() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = 1024;
    let boundaries = vec![
        MatrixBoundarySpec::new(n, (0..n).collect(), (0..n).collect())
            .unwrap()
            .with_protected_inputs(vec![0, 63, n - 1])
            .unwrap()
            .with_zero_outputs(vec![0, n - 1])
            .unwrap(),
    ];
    let mut registry = FeatureRegistry::new();
    register_matrix_boundaries(&mut registry, boundaries.clone(), 4096, 4096).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: [(b'C', 8), (b'I', 2), (b'I', 3)]
            .map(|(f, k)| FeatureId::new(f, k).unwrap())
            .to_vec(),
        seed: 0,
        schema_id: "wide-boundary-gpu".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        n,
        0,
        DeviceFeatureContextInputs {
            matrix_boundaries: Some(MatrixBoundaryPlanInputs {
                boundaries: &boundaries,
                max_ports: 4096,
                max_equations: 4096,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let identity = MatrixState::identity(n, "boundary-large").unwrap();
    let mut changed = identity.clone();
    for (control, target) in [(n / 2, 0), (1, n - 1), (n - 1, 63)] {
        changed.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, changed];
    let raw: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect();
    let rows = Tensor::from_slice(&raw)
        .view([2, i64::from(n), i64::from(n.div_ceil(64))])
        .to_device(Device::Cuda(0));
    let actual = plan
        .build(&DeviceStateBatch::from_packed_tensor(rows, n, 0).unwrap())
        .unwrap();
    let reference = registry.build(&states, &policy).unwrap();
    for &id in &policy.enabled_ids {
        let field = actual.feature(id).unwrap();
        let expected = reference.feature(id);
        assert_eq!(expected.status, [FeatureStatus::Computed; 2]);
        assert_eq!(
            Vec::<u8>::try_from(field.status().to_device(Device::Cpu)).unwrap(),
            [1, 1]
        );
        assert_eq!(flat(field.values()), expected.values, "{id} values");
        if let Some(mask) = field.token_valid() {
            assert_eq!(
                Vec::<u8>::try_from(mask.to_device(Device::Cpu).view([-1])).unwrap(),
                *expected.token_valid.as_ref().unwrap()
            );
            assert_eq!(
                flat(field.incidence().unwrap()),
                *expected.incidence.as_ref().unwrap(),
                "{id} incidence"
            );
        }
    }
    println!("wide boundary n={n}, batch=2, 1026 equations x 32 packed words per state; all C08/I02/I03 values, token masks and incidence match scalar");
}
