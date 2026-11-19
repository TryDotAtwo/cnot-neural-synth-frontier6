use super::{all, ints, resident};
use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, GuardedPlanInputs,
};
use cnot103_native::{
    CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState,
};
use tch::{Device, Kind, Tensor};

fn inputs(max_n: u32, max_entries: usize) -> DeviceFeatureContextInputs<'static> {
    DeviceFeatureContextInputs {
        guarded: Some(GuardedPlanInputs { max_n, max_entries }),
        ..Default::default()
    }
}
fn policy(with_zero: bool) -> FeaturePolicy {
    let mut enabled_ids = vec![FeatureId::new(b'F', 8).unwrap()];
    if with_zero {
        enabled_ids.push(FeatureId::new(b'F', 7).unwrap());
    }
    FeaturePolicy {
        enabled_ids,
        seed: 0,
        schema_id: "one-excess-gpu-v1".into(),
        solver_assisted: true,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn registry(max_n: u32, max_entries: usize) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    cnot103_native::register_guarded_one_excess(&mut r, max_n, max_entries).unwrap();
    cnot103_native::register_guarded_zero_excess_with_budget(&mut r, max_n, max_entries).unwrap();
    r
}
fn flat(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Float).flatten(0, -1)).unwrap()
}
fn compare(states: &[MatrixState], a: &CnotAlphabet, max_n: u32, budget: usize, with_zero: bool) {
    let gpu = resident(states, a);
    let r = registry(max_n, budget);
    let p = policy(with_zero);
    let expected = r.build(states, &p).unwrap();
    let packet = DeviceFeaturePlan::with_contexts(&r, &p, a.dimension(), 0, inputs(max_n, budget))
        .unwrap()
        .build(&gpu)
        .unwrap();
    for id in &p.enabled_ids {
        let e = expected.feature(*id);
        let actual = packet.feature(*id).unwrap();
        let want: Vec<_> = e
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                _ => panic!("unexpected status"),
            })
            .collect();
        assert_eq!(ints(actual.status()), want);
        assert_eq!(flat(actual.values()), e.values);
    }
}
#[test]
fn gpu_one_excess_feature_plan_preserves_statuses_and_shared_prerequisite() {
    let (a, states) = all(3);
    for with_zero in [false, true] {
        for (max_n, budget) in [(2, 100_000), (3, 16), (3, 27), (3, 100_000)] {
            compare(&states, &a, max_n, budget, with_zero);
        }
        let a = CnotAlphabet::all_to_all_unit(8).unwrap();
        let s = MatrixState::from_packed_rows(8, a.id(), vec![17, 130, 128, 40, 88, 160, 104, 132])
            .unwrap();
        compare(&[s], &a, 8, 729, with_zero);
        let custom = CnotAlphabet::new(
            3,
            "one-excess-plan-custom",
            vec![CnotGate {
                control: 0,
                target: 1,
            }],
            vec![2],
        )
        .unwrap();
        let state = MatrixState::identity(3, custom.id()).unwrap();
        compare(&[state.clone()], &custom, 2, 17, with_zero);
        compare(&[state], &custom, 3, 17, with_zero);
    }
}
#[test]
fn gpu_one_excess_feature_plan_requires_exact_producer_policy_and_limits() {
    let r = registry(3, 17);
    let mut p = policy(false);
    for input in [inputs(4, 17), inputs(3, 18), Default::default()] {
        assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, input).is_err());
    }
    p.solver_assisted = false;
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17)).is_err());
    p.solver_assisted = true;
    for symmetry in ["exact_relabel", "augmentation_only"] {
        p.symmetry_contract = symmetry.into();
        assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17)).is_ok());
    }
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let identity = MatrixState::identity(3, a.id()).unwrap();
    let states = [identity.clone(), identity];
    let packed = resident(&states, &a);
    for (bytes, evaluations) in [(Some(1), None), (None, Some(1))] {
        p.max_bytes = bytes;
        p.evaluations = evaluations;
        let expected = r.build(&states, &p).unwrap();
        let actual = DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17))
            .unwrap()
            .build(&packed)
            .unwrap();
        let id = p.enabled_ids[0];
        let reference = expected.feature(id);
        assert_eq!(reference.status[1], FeatureStatus::ResourceLimit);
        let codes: Vec<_> = reference
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                _ => panic!("unexpected F08 status"),
            })
            .collect();
        let entry = actual.feature(id).unwrap();
        assert_eq!(ints(entry.status()), codes);
        assert_eq!(flat(entry.values()), reference.values);
    }
    p.evaluations = None;
    p.enabled_ids.push(FeatureId::new(b'G', 1).unwrap());
    assert!(DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(3, 17)).is_err());
}
#[test]
fn gpu_one_excess_feature_plan_preserves_all_model_heads_and_named_gradients() {
    use cnot103_native::model::{FieldOutput, FusionDistanceField, ModelConfig};
    use tch::nn::VarStore;
    let (a, states) = all(3);
    let samples = [
        states
            .iter()
            .find(|s| {
                matches!(
                    cnot103_native::recognize_stage38(s, &a, 100_000).unwrap(),
                    cnot103_native::Stage38Result::Exact(_)
                )
            })
            .unwrap()
            .clone(),
        states
            .iter()
            .find(|s| {
                matches!(
                    cnot103_native::recognize_stage38(s, &a, 100_000).unwrap(),
                    cnot103_native::Stage38Result::CertifiedNoSolutionWithinScope(_)
                )
            })
            .unwrap()
            .clone(),
        MatrixState::identity(3, a.id()).unwrap(),
        MatrixState::from_packed_rows(3, a.id(), vec![3, 5, 1]).unwrap(),
    ];
    let gpu = resident(&samples, &a);
    for (max_n, budget) in [(2, 100_000), (3, 16), (3, 100_000)] {
        let r = registry(max_n, budget);
        let mut p = policy(true);
        p.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let actual = DeviceFeaturePlan::with_contexts(&r, &p, 3, 0, inputs(max_n, budget))
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
        assert_eq!(ints(&x.action_mask), ints(&y.action_mask));
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
