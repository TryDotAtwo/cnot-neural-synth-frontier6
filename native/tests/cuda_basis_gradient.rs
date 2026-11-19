use cnot103_native::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    register_basis_gradient, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus,
    MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn flat(x: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&x.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn check(n: u32, ordinals: &[u8]) {
    let mut registry = FeatureRegistry::new();
    register_basis_gradient(&mut registry).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: std::iter::once(FeatureId::new(b'A', 1).unwrap())
            .chain(ordinals.iter().map(|&i| FeatureId::new(b'E', i).unwrap()))
            .collect(),
        seed: 0,
        schema_id: "gpu-basis-gradient-v1".into(),
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
        DeviceFeatureContextInputs::default(),
    )
    .unwrap();
    let identity = MatrixState::identity(n, "basis-gradient").unwrap();
    let width = n / 2;
    let mut shear = identity.clone();
    for target in width..n {
        for control in 0..width.min(2) {
            shear.apply_cnot(CnotGate { control, target }).unwrap();
        }
    }
    let mut transformed = shear.clone();
    if width > 1 {
        transformed
            .apply_cnot(CnotGate {
                control: 0,
                target: 1,
            })
            .unwrap();
    }
    if width > 64 {
        transformed
            .apply_cnot(CnotGate {
                control: 66,
                target: width,
            })
            .unwrap();
        transformed
            .apply_cnot(CnotGate {
                control: 1,
                target: 66,
            })
            .unwrap();
    }
    let mut moved = transformed.clone();
    for (control, target) in [(0, n - 1), (n - 1, 0), (0, n - 1)] {
        moved.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let mut mixed = identity.clone();
    for step in 0..3 * n {
        let control = (17 * step + 1) % n;
        let mut target = (29 * step + 2) % n;
        if target == control {
            target = (target + 1) % n;
        }
        mixed.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, shear, transformed, moved, mixed];
    let words: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&w| w as i64))
        .collect();
    let device_states = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&words)
            .view([5, i64::from(n), i64::from(n.div_ceil(64))])
            .to_device(Device::Cuda(0)),
        n,
        0,
    )
    .unwrap();
    let actual = plan.build(&device_states).unwrap();
    let expected = registry.build(&states, &policy).unwrap();
    for &id in &policy.enabled_ids {
        let a = actual.feature(id).unwrap();
        let e = expected.feature(id);
        assert_eq!(a.spec(), &e.spec);
        assert_eq!(a.values().device(), Device::Cuda(0));
        let status: Vec<u8> = e
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Inapplicable => 2,
                FeatureStatus::Unsupported => 3,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            Vec::<u8>::try_from(&a.status().to_device(Device::Cpu)).unwrap(),
            status,
            "{id} n={n}"
        );
        assert_eq!(flat(a.values()), e.values, "{id} values n={n}");
    }
    if n == 6 && ordinals == [1, 4, 5] {
        assert_eq!(
            flat(
                &actual
                    .feature(FeatureId::new(b'E', 1).unwrap())
                    .unwrap()
                    .values()
                    .get(1)
            ),
            vec![6.0]
        );
        assert_eq!(
            flat(
                &actual
                    .feature(FeatureId::new(b'E', 5).unwrap())
                    .unwrap()
                    .values()
                    .get(1)
            ),
            vec![-1.0, 2.0, 5.0, 1.0 / 3.0]
        );
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
    let a = model.forward_device(&actual).unwrap();
    let e = model.forward(&expected).unwrap();
    assert!(a.value.allclose(&e.value, 3e-5, 3e-5, false));
    assert!(a.qhat.masked_select(&a.action_mask).allclose(
        &e.qhat.masked_select(&e.action_mask),
        3e-5,
        3e-5,
        false
    ));
    assert!(a.proposal_logits.masked_select(&a.action_mask).allclose(
        &e.proposal_logits.masked_select(&e.action_mask),
        3e-5,
        3e-5,
        false
    ));
    assert_eq!(a.value.double_value(&[0]), 0.0);
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
    loss(&a).backward();
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
    loss(&e).backward();
    for (name, v) in vars.variables() {
        let g = v.grad();
        let g = if g.defined() {
            g
        } else {
            Tensor::zeros_like(&v)
        };
        assert!(
            gradients[&name].allclose(&g, 5e-5, 5e-5, false),
            "gradient {name} n={n}"
        );
    }
}

#[test]
fn cuda_basis_gradient_matches_source_values_statuses_heads_and_gradients() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    for n in [2, 3, 4, 6, 65, 144] {
        check(n, &[1, 4, 5]);
    }
}

#[test]
fn cuda_basis_gradient_handles_individually_selected_features() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    for ordinal in [1, 4, 5] {
        check(6, &[ordinal]);
    }
}
