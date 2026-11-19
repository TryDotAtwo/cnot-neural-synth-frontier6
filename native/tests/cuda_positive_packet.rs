use cnot103_native::device_feature_plan::{
    DeviceFeatureContextInputs, DeviceFeaturePlan, PositivePacketPlanInputs,
};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    register_positive_packet_contexts, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureStatus, LocalBooleanFactor, MatrixState, OptionalGatePacket,
};
use num_bigint::BigUint;
use tch::{nn::VarStore, Device, Kind, Tensor};

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'I', 8).unwrap()],
        seed: 0,
        schema_id: "cuda-positive-packet-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}
fn inputs(contexts: &[OptionalGatePacket], limits: [usize; 4]) -> DeviceFeatureContextInputs<'_> {
    DeviceFeatureContextInputs {
        positive_packet: Some(PositivePacketPlanInputs {
            contexts,
            max_n: limits[0],
            max_slots: limits[1],
            max_monomials: limits[2],
            max_factor_entries: limits[3],
        }),
        ..Default::default()
    }
}
fn states(alphabet: &CnotAlphabet) -> Vec<MatrixState> {
    let identity = MatrixState::identity(alphabet.dimension(), alphabet.id()).unwrap();
    let mut changed = identity.clone();
    changed
        .apply_cnot(CnotGate {
            control: 0,
            target: alphabet.dimension() - 1,
        })
        .unwrap();
    vec![identity, changed]
}
fn resident(states: &[MatrixState], alphabet: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = alphabet.dimension();
    let packed: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
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
fn registry(contexts: &[OptionalGatePacket], limits: [usize; 4]) -> FeatureRegistry {
    let mut r = FeatureRegistry::new();
    register_positive_packet_contexts(
        &mut r,
        contexts.to_vec(),
        limits[0],
        limits[1],
        limits[2],
        limits[3],
    )
    .unwrap();
    r
}
fn compare(
    alphabet: &CnotAlphabet,
    contexts: &[OptionalGatePacket],
    limits: [usize; 4],
) -> (Vec<f32>, Vec<u8>) {
    let states = states(alphabet);
    let gpu = resident(&states, alphabet);
    let registry = registry(contexts, limits);
    let policy = policy();
    let plan = DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        alphabet.dimension(),
        0,
        inputs(contexts, limits),
    )
    .unwrap();
    let actual = plan.build(&gpu).unwrap();
    let expected = registry.build(&states, &policy).unwrap();
    let id = FeatureId::new(b'I', 8).unwrap();
    let a = actual.feature(id).unwrap();
    let e = expected.feature(id);
    assert_eq!(a.spec(), &e.spec);
    assert_eq!(a.values().device(), Device::Cuda(0));
    let g = e.shape[1] as i64;
    let cap = a.values().size()[1];
    assert!(cap >= g);
    let values = flat(&a.values().narrow(1, 0, g));
    assert_eq!(values, e.values);
    assert_eq!(
        flat(&a.incidence().unwrap().narrow(1, 0, g)),
        *e.incidence.as_ref().unwrap()
    );
    assert_eq!(
        Vec::<u8>::try_from(
            &a.token_valid()
                .unwrap()
                .narrow(1, 0, g)
                .to_kind(Kind::Uint8)
                .to_device(Device::Cpu)
                .contiguous()
                .view([-1])
        )
        .unwrap(),
        *e.token_valid.as_ref().unwrap()
    );
    for tensor in [a.values(), a.token_valid().unwrap(), a.incidence().unwrap()] {
        assert!(flat(&tensor.narrow(1, g, cap - g).to_kind(Kind::Float))
            .iter()
            .all(|&v| v == 0.0));
    }
    let statuses: Vec<u8> = e
        .status
        .iter()
        .map(|s| match s {
            FeatureStatus::Computed => 1,
            FeatureStatus::Unsupported => 3,
            FeatureStatus::ResourceLimit => 4,
            s => panic!("unexpected {s:?}"),
        })
        .collect();
    assert_eq!(
        Vec::<u8>::try_from(&a.status().to_device(Device::Cpu)).unwrap(),
        statuses
    );
    (values, statuses)
}
#[test]
fn cuda_positive_packet_matches_stage59_literal_cut_values_and_full_incidence() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let context = OptionalGatePacket::new(
        3,
        vec![(0, 2), (1, 2), (0, 2), (1, 2)],
        vec![
            LocalBooleanFactor::new(vec![0, 3], vec![BigUint::from(1u8), BigUint::from(2u8)])
                .unwrap(),
        ],
    )
    .unwrap();
    let (values, status) = compare(&alphabet, &[context], [144, 256, 4096, 65_536]);
    assert_eq!(status, [1, 1]);
    assert_eq!(
        &values[..20],
        &[0., 0., 0., 0., 1., 1., 1., 2., 2., 2., 1., 3., 3., 1., 1., 2., 4., 0., 0., 0.]
    );
}
#[test]
fn cuda_positive_packet_matches_all_three_wire_packets_up_to_four_slots() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let choices: Vec<_> = alphabet
        .gates()
        .iter()
        .map(|g| (g.control, g.target))
        .collect();
    let mut packets = 0;
    for len in 0..=4u32 {
        for encoded in 0..6usize.pow(len) {
            let mut number = encoded;
            let gates: Vec<_> = (0..len)
                .map(|_| {
                    let gate = choices[number % 6];
                    number /= 6;
                    gate
                })
                .collect();
            let factors = if len > 1 {
                vec![LocalBooleanFactor::new(vec![0, len - 1], vec![]).unwrap()]
            } else {
                vec![]
            };
            let context = OptionalGatePacket::new(3, gates, factors).unwrap();
            compare(&alphabet, &[context.clone()], [3, 4, 1usize << len, 16]);
            if len > 1 {
                compare(&alphabet, &[context], [3, 4, len as usize + 1, 16]);
            }
            packets += 1;
        }
    }
    assert_eq!(packets, 1555);
}
#[test]
fn cuda_positive_packet_preserves_wide_scopes_empty_packets_and_repeated_physical_roles() {
    for n in [2, 65, 144] {
        let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
        let empty = OptionalGatePacket::new(
            n,
            vec![],
            vec![LocalBooleanFactor::new(vec![], vec![]).unwrap()],
        )
        .unwrap();
        assert_eq!(
            compare(&alphabet, &[empty], [144, 256, 1, 65_536]).1,
            [1, 1]
        );
        let factor =
            LocalBooleanFactor::new(vec![69, 70], vec![BigUint::from(1u8), BigUint::from(2u8)])
                .unwrap();
        let c = OptionalGatePacket::new(n, vec![(0, n - 1); 71], vec![factor]).unwrap();
        let (values, status) = compare(&alphabet, &[c], [144, 256, 72, 65_536]);
        assert_eq!(status, [1, 1]);
        assert_eq!(&values[70 * 4..71 * 4], &[70., 1., 1., 2.]);
        let factor = LocalBooleanFactor::new(
            (0..130).rev().collect(),
            vec![BigUint::from(1u8) << 129usize],
        )
        .unwrap();
        let gates = (0..130)
            .map(|j| ((if n > 2 { j % 2 } else { 0 }) as u32, n - 1))
            .collect();
        let c = OptionalGatePacket::new(n, gates, vec![factor]).unwrap();
        let (values, status) = compare(&alphabet, &[c], [144, 256, 131, 65_536]);
        assert_eq!(status, [1, 1]);
        assert_eq!(values[129 * 4 + 2], 129.);
        assert_eq!(&values[130 * 4..131 * 4], &[130., 0., 0., 0.]);
    }
}

#[test]
fn cuda_positive_packet_preserves_budget_alphabet_and_nonaffine_priority() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let wrong = CnotAlphabet::new(3, "i08-wrong", alphabet.gates().to_vec(), vec![1; 6]).unwrap();
    let affine = OptionalGatePacket::new(
        3,
        vec![(0, 2), (1, 2), (0, 2), (1, 2)],
        vec![
            LocalBooleanFactor::new(vec![0, 3], vec![BigUint::from(1u8), BigUint::from(2u8)])
                .unwrap(),
        ],
    )
    .unwrap();
    for limits in [
        [2, 256, 4096, 65_536],
        [144, 3, 4096, 65_536],
        [144, 256, 4096, 3],
    ] {
        for alpha in [&alphabet, &wrong] {
            assert_eq!(compare(alpha, &[affine.clone()], limits).1, [4, 4]);
        }
    }
    for monomials in [1, 4, 500_000, usize::MAX] {
        assert_eq!(
            compare(&alphabet, &[affine.clone()], [144, 256, monomials, 65_536]).1,
            [4, 4]
        );
        assert_eq!(
            compare(&wrong, &[affine.clone()], [144, 256, monomials, 65_536]).1,
            [3, 3]
        );
    }
    assert_eq!(
        compare(&alphabet, &[affine], [144, 256, 5, 65_536]).1,
        [1, 1]
    );
    let nonlinear = OptionalGatePacket::new(3, vec![(0, 1), (1, 2), (2, 0)], vec![]).unwrap();
    // The first two gates already prove nonaffinity. Later terms must still be
    // counted, so cap4 yields ResourceLimit while cap7 yields Unsupported.
    assert_eq!(
        compare(&alphabet, &[nonlinear.clone()], [144, 256, 4, 65_536]).1,
        [4, 4]
    );
    assert_eq!(
        compare(&alphabet, &[nonlinear], [144, 256, 7, 65_536]).1,
        [3, 3]
    );
    let missing = OptionalGatePacket::new(2, vec![(0, 1)], vec![]).unwrap();
    assert_eq!(
        compare(&alphabet, &[missing], [2, 256, 4096, 65_536]).1,
        [3, 3]
    );
}

#[test]
fn cuda_positive_packet_binds_full_factor_truth_tables_and_rejects_missing_identity() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let contexts = [OptionalGatePacket::new(
        3,
        vec![(0, 2), (1, 2)],
        vec![LocalBooleanFactor::new(vec![0, 1], vec![]).unwrap()],
    )
    .unwrap()];
    let limits = [144, 256, 4096, 65_536];
    let registry = registry(&contexts, limits);
    let policy = policy();
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        3,
        0,
        DeviceFeatureContextInputs::default()
    )
    .err()
    .unwrap()
    .contains("explicit"));
    assert!(DeviceFeaturePlan::with_contexts(
        &registry,
        &policy,
        3,
        0,
        inputs(&contexts, [143, 256, 4096, 65_536])
    )
    .err()
    .unwrap()
    .contains("metadata"));
    for factor in [
        LocalBooleanFactor::new(vec![1, 0], vec![]).unwrap(),
        LocalBooleanFactor::new(vec![0, 1], vec![BigUint::from(0u8)]).unwrap(),
    ] {
        let changed =
            [OptionalGatePacket::new(3, contexts[0].gates().to_vec(), vec![factor]).unwrap()];
        assert!(DeviceFeaturePlan::with_contexts(
            &registry,
            &policy,
            3,
            0,
            inputs(&changed, limits)
        )
        .err()
        .unwrap()
        .contains("metadata"));
    }
    let plan =
        DeviceFeaturePlan::with_contexts(&registry, &policy, 3, 0, inputs(&contexts, limits))
            .unwrap();
    let gpu = resident(&states(&alphabet), &alphabet);
    let unlabelled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 3, 0).unwrap();
    assert!(plan.build(&unlabelled).err().unwrap().contains("alphabet"));
    let mut exact = policy;
    exact.symmetry_contract = "exact_relabel".into();
    assert!(
        DeviceFeaturePlan::with_contexts(&registry, &exact, 3, 0, inputs(&contexts, limits))
            .is_err()
    );
}

#[test]
fn cuda_positive_packet_preserves_all_model_heads_gradients_and_masked_unavailable_groups() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let states = states(&alphabet);
    let gpu = resident(&states, &alphabet);
    for (gates, budget) in [
        (vec![], 1),
        (vec![(0, 2), (1, 2), (0, 2)], 4),
        (vec![(0, 1), (1, 2)], 4),
        (vec![(0, 1), (1, 2)], 1),
    ] {
        let contexts = [OptionalGatePacket::new(3, gates, vec![]).unwrap()];
        let limits = [144, 256, budget, 65_536];
        let registry = registry(&contexts, limits);
        let mut policy = policy();
        policy.enabled_ids.push(FeatureId::new(b'A', 1).unwrap());
        let plan =
            DeviceFeaturePlan::with_contexts(&registry, &policy, 3, 0, inputs(&contexts, limits))
                .unwrap();
        let actual = plan.build(&gpu).unwrap();
        let expected = registry.build(&states, &policy).unwrap();
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
