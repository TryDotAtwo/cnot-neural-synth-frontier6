//! Exercise canonical feature-major/row-minor budget spending against the
//! independent scalar registry. Only tests download payloads or group extents.
use crate::{
    device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan},
    device_state::DeviceStateBatch,
    model::DeviceFeaturePacket,
    run_config::parse_run_manifest,
    CnotGate, FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState,
};
use std::{fs, path::PathBuf};
use tch::{Device, Kind, Tensor};

#[test]
#[ignore = "requires CUDA and the pinned n8 training manifest in CNOT103_F08_RUN"]
fn cached_root_f08_matches_scalar_packets_with_missing_status() {
    use crate::{model::DeviceFeatureEntry, CnotAlphabet, FeatureId};
    let path = std::env::var_os("CNOT103_F08_RUN").expect("pinned n8 manifest");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    value["feature_contexts"]["f08_max_history_counts"] = serde_json::json!(1);
    let manifest = parse_run_manifest(&serde_json::to_vec(&value).unwrap()).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(8).unwrap();
    let runtime = super::build(manifest.feature_contexts, manifest.feature_policy,
        std::slice::from_ref(&alphabet), manifest.training.seed, [8].into_iter().collect(), 0).unwrap();
    let identity = MatrixState::identity(8, alphabet.id()).unwrap();
    let mut swap = identity.clone();
    for (control, target) in [(1, 0), (0, 1), (1, 0)] {
        swap.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, swap.clone(), swap];
    let raw: Vec<_> = states.iter().flat_map(|s| s.packed_rows().iter().map(|&w| w as i64)).collect();
    let packed = DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&raw).view([3, 8, 1]).to_device(Device::Cuda(0)), &alphabet, 0,
    ).unwrap();
    let plan = &runtime.plans[&8];
    let f08 = FeatureId::new(b'F', 8).unwrap();
    let batched = plan.build(&packed).unwrap();
    let feature = batched.feature(f08).unwrap();
    assert_eq!(Vec::<u8>::try_from(feature.status().to_device(Device::Cpu)).unwrap(), vec![1, 4, 4]);
    for row in 0..3 {
        let root = packed.row(row).unwrap();
        let reference = plan.build(&root).unwrap();
        let cached = DeviceFeatureEntry::new(feature.spec().clone(),
            feature.values().narrow(0, row, 1), feature.status().narrow(0, row, 1), None, None);
        let actual = plan.build_with_root_f08(&root, Some(&cached)).unwrap();
        for &id in &runtime.policy.enabled_ids {
            let left = reference.feature(id).unwrap();
            let right = actual.feature(id).unwrap();
            assert!(left.status().equal(right.status()), "{id} status");
            assert!(left.values().equal(right.values()), "{id} values");
            match (left.token_valid(), right.token_valid()) {
                (Some(a), Some(b)) => assert!(a.equal(b), "{id} token mask"),
                (None, None) => {},
                _ => panic!("{id} token mask presence"),
            }
            match (left.incidence(), right.incidence()) {
                (Some(a), Some(b)) => assert!(a.equal(b), "{id} incidence"),
                (None, None) => {},
                _ => panic!("{id} incidence presence"),
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and the pinned n8 training manifest in CNOT103_F08_RUN"]
fn f08_history_call_guard_limits_work_and_matches_cpu() {
    use crate::{CnotAlphabet, FeatureId};
    let path = std::env::var_os("CNOT103_F08_RUN").expect("pinned n8 manifest");
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(8).unwrap();
    let runtime = |cap: Option<u32>| {
        let mut run = value.clone();
        if let Some(cap) = cap {
            run["feature_contexts"]["f08_max_history_counts"] = serde_json::json!(cap);
        }
        let m = parse_run_manifest(&serde_json::to_vec(&run).unwrap()).unwrap();
        super::build(m.feature_contexts, m.feature_policy, std::slice::from_ref(&alphabet),
                     m.training.seed, [8].into_iter().collect(), 0).unwrap()
    };
    let unrestricted = runtime(None);
    let cap_one = runtime(Some(1));
    let cap_large = runtime(Some(1000));
    let f08 = FeatureId::new(b'F', 8).unwrap();
    let identity = MatrixState::identity(8, alphabet.id()).unwrap();
    let mut swap = identity.clone();
    for (control, target) in [(1, 0), (0, 1), (1, 0)] {
        swap.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, swap.clone(), swap];
    let raw: Vec<_> = states.iter().flat_map(|s| s.packed_rows().iter().map(|&w| w as i64)).collect();
    let packed = DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&raw).view([3, 8, 1]).to_device(Device::Cuda(0)), &alphabet, 0,
    ).unwrap();
    for (runtime, expected) in [(&unrestricted, vec![1, 1, 1]),
                                (&cap_one, vec![1, 4, 4]),
                                (&cap_large, vec![1, 1, 1])] {
        let cpu = runtime.registry.build(&states, &runtime.policy).unwrap();
        let gpu = runtime.plans[&8].build(&packed).unwrap();
        compare(&gpu, &cpu, &runtime.policy);
        assert_eq!(Vec::<u8>::try_from(gpu.feature(f08).unwrap().status().to_device(Device::Cpu)).unwrap(), expected);
    }
}

#[test]
#[ignore = "requires CUDA and the pinned n8 training manifest in CNOT103_F08_RUN"]
fn f08_component_guard_retains_disconnected_large_examples_and_matches_cpu() {
    use crate::{CnotAlphabet, FeatureId};
    let path = std::env::var_os("CNOT103_F08_RUN").expect("pinned n8 manifest");
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(8).unwrap();
    let runtime = |cap: Option<u32>| {
        let mut run = value.clone();
        if let Some(cap) = cap {
            run["feature_contexts"]["f08_max_component_events"] = serde_json::json!(cap);
        }
        let m = parse_run_manifest(&serde_json::to_vec(&run).unwrap()).unwrap();
        super::build(m.feature_contexts, m.feature_policy, std::slice::from_ref(&alphabet),
                     m.training.seed, [8].into_iter().collect(), 0).unwrap()
    };
    let unrestricted = runtime(None);
    let cap_two = runtime(Some(2));
    let cap_three = runtime(Some(3));
    let f08 = FeatureId::new(b'F', 8).unwrap();
    let identity = MatrixState::identity(8, alphabet.id()).unwrap();
    let mut swap = identity.clone();
    for (control, target) in [(1, 0), (0, 1), (1, 0)] {
        swap.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let mut disconnected = swap.clone();
    for (control, target) in [(2, 3), (4, 5), (6, 7)] {
        disconnected.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, swap, disconnected.clone(), disconnected];
    let raw: Vec<_> = states.iter().flat_map(|s| s.packed_rows().iter().map(|&w| w as i64)).collect();
    let packed = DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&raw).view([4, 8, 1]).to_device(Device::Cuda(0)), &alphabet, 0,
    ).unwrap();
    for (runtime, expected) in [(&unrestricted, vec![1, 1, 1, 1]),
                                (&cap_two, vec![1, 4, 4, 4]),
                                (&cap_three, vec![1, 1, 1, 1])] {
        let cpu = runtime.registry.build(&states, &runtime.policy).unwrap();
        let gpu = runtime.plans[&8].build(&packed).unwrap();
        compare(&gpu, &cpu, &runtime.policy);
        assert_eq!(Vec::<u8>::try_from(gpu.feature(f08).unwrap().status().to_device(Device::Cpu)).unwrap(), expected);
    }
}

#[test]
#[ignore = "requires CUDA and the pinned n8 training manifest in CNOT103_F08_RUN"]
fn f08_changed_rows_guard_keeps_easy_large_cases_and_matches_cpu() {
    use crate::{CnotAlphabet, FeatureId};
    let path = std::env::var_os("CNOT103_F08_RUN").expect("pinned n8 manifest");
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(8).unwrap();
    let runtime = |cap: Option<u32>| {
        let mut run = value.clone();
        if let Some(cap) = cap {
            run["feature_contexts"]["f08_max_changed_rows"] = serde_json::json!(cap);
        }
        let m = parse_run_manifest(&serde_json::to_vec(&run).unwrap()).unwrap();
        super::build(m.feature_contexts, m.feature_policy, std::slice::from_ref(&alphabet),
                     m.training.seed, [8].into_iter().collect(), 0).unwrap()
    };
    let unrestricted = runtime(None);
    let cap_one = runtime(Some(1));
    let cap_two = runtime(Some(2));
    let f08 = FeatureId::new(b'F', 8).unwrap();
    for (a, b) in unrestricted.registry.specs().iter().zip(cap_one.registry.specs()) {
        if a.id == f08 { assert_ne!(a.producer_version, b.producer_version); }
        else { assert_eq!(a, &b, "{} contract changed", a.id); }
    }
    let identity = MatrixState::identity(8, alphabet.id()).unwrap();
    let mut easy = identity.clone();
    for (control, target) in [(0, 1), (2, 3)] {
        easy.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let mut swap = identity.clone();
    for (control, target) in [(1, 0), (0, 1), (1, 0)] {
        swap.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [identity, easy, swap.clone(), swap];
    let raw: Vec<_> = states.iter().flat_map(|s| s.packed_rows().iter().map(|&w| w as i64)).collect();
    let packed = DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&raw).view([4, 8, 1]).to_device(Device::Cuda(0)), &alphabet, 0,
    ).unwrap();
    for (runtime, expected) in [(&unrestricted, vec![1, 1, 1, 1]),
                                (&cap_one, vec![1, 1, 4, 4]),
                                (&cap_two, vec![1, 1, 1, 1])] {
        let cpu = runtime.registry.build(&states, &runtime.policy).unwrap();
        let gpu = runtime.plans[&8].build(&packed).unwrap();
        compare(&gpu, &cpu, &runtime.policy);
        let f = gpu.feature(f08).unwrap();
        assert_eq!(Vec::<u8>::try_from(f.status().to_device(Device::Cpu)).unwrap(), expected);
        if expected[2] == 4 {
            assert_eq!(f.values().get(2).abs().sum(Kind::Float).double_value(&[]), 0.0);
        }
    }
}

#[test]
#[ignore = "requires CUDA and the pinned n8 training manifest in CNOT103_F08_RUN"]
fn f08_limit_preserves_other_inputs_and_trains_the_missing_token() {
    use crate::model::{FusionDistanceField, ModelConfig};
    use crate::{CnotAlphabet, FeatureId};
    use tch::nn::VarStore;

    let path = std::env::var_os("CNOT103_F08_RUN").expect("pinned n8 manifest");
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(8).unwrap();
    let runtime = |cap: Option<u32>| {
        let mut run = value.clone();
        if let Some(cap) = cap {
            run["feature_contexts"]["f08_max_n"] = serde_json::json!(cap);
        }
        let m = parse_run_manifest(&serde_json::to_vec(&run).unwrap()).unwrap();
        super::build(
            m.feature_contexts,
            m.feature_policy,
            std::slice::from_ref(&alphabet),
            m.training.seed,
            [8].into_iter().collect(),
            0,
        )
        .unwrap()
    };
    let baseline = runtime(None);
    let limited = runtime(Some(2));
    assert_ne!(
        baseline.registry.contract_sha256(),
        limited.registry.contract_sha256()
    );

    let identity = MatrixState::identity(8, alphabet.id()).unwrap();
    let mut adjacent = identity.clone();
    adjacent
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let raw: Vec<_> = [identity, adjacent]
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&w| w as i64))
        .collect();
    let packed = DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&raw)
            .view([2, 8, 1])
            .to_device(Device::Cuda(0)),
        &alphabet,
        0,
    )
    .unwrap();
    let normal_packet = baseline.plans[&8].build(&packed).unwrap();
    let limited_packet = limited.plans[&8].build(&packed).unwrap();
    let f08 = FeatureId::new(b'F', 8).unwrap();
    for id in &limited.policy.enabled_ids {
        let a = normal_packet.feature(*id).unwrap();
        let b = limited_packet.feature(*id).unwrap();
        if *id == f08 {
            assert_eq!(
                Vec::<u8>::try_from(a.status().to_device(Device::Cpu)).unwrap(),
                vec![1, 1]
            );
            assert_eq!(
                Vec::<u8>::try_from(b.status().to_device(Device::Cpu)).unwrap(),
                vec![4, 4]
            );
        } else {
            assert!(a.status().equal(b.status()), "{id} status changed");
            assert!(
                a.values().allclose(b.values(), 1e-6, 1e-6, false),
                "{id} values changed"
            );
            match (a.token_valid(), b.token_valid()) {
                (Some(x), Some(y)) => assert!(x.equal(y), "{id} mask changed"),
                (None, None) => (),
                _ => panic!("{id} mask layout changed"),
            }
            match (a.incidence(), b.incidence()) {
                (Some(x), Some(y)) => {
                    assert!(x.allclose(y, 1e-6, 1e-6, false), "{id} incidence changed")
                }
                (None, None) => (),
                _ => panic!("{id} incidence layout changed"),
            }
        }
    }

    tch::manual_seed(1497);
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &limited.registry,
        &limited.policy,
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
    let output = model.forward_device(&limited_packet).unwrap();
    assert!(output.value.isfinite().all().int64_value(&[]) != 0);
    output.value.sum(Kind::Float).backward();
    let specs = limited.registry.selected_specs(&limited.policy).unwrap();
    let index = specs.iter().position(|s| s.id == f08).unwrap();
    let weight = vars
        .variables()
        .remove("stem.missing_status.weight")
        .unwrap();
    let column = weight.grad().select(1, (index * 3 + 2) as i64);
    let magnitude = column.abs().sum(Kind::Float).double_value(&[]);
    assert!(
        magnitude.is_finite() && magnitude > 0.0,
        "F08 resource token gradient"
    );
}

fn compare(gpu: &DeviceFeaturePacket, cpu: &crate::FeaturePacket, policy: &FeaturePolicy) {
    let widths = Vec::<i64>::try_from(gpu.source_group_widths().to_device(Device::Cpu)).unwrap();
    let gpu = gpu.source_layout(&widths).unwrap();
    for &id in &policy.enabled_ids {
        let actual = gpu.feature(id).unwrap();
        let expected = cpu.feature(id);
        let codes: Vec<u8> = expected
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::Inapplicable => 2,
                FeatureStatus::Unsupported => 3,
                FeatureStatus::ResourceLimit => 4,
                FeatureStatus::NotRequested => 0,
            })
            .collect();
        assert_eq!(
            Vec::<u8>::try_from(actual.status().to_device(Device::Cpu)).unwrap(),
            codes,
            "{id} status"
        );
        let shape: Vec<i64> = expected.shape.iter().map(|&x| x as i64).collect();
        assert_eq!(actual.values().size(), shape, "{id} shape");
        assert!(
            actual.values().to_device(Device::Cpu).view([-1]).allclose(
                &Tensor::from_slice(&expected.values),
                2e-5,
                2e-5,
                false
            ),
            "{id} values"
        );
        if let Some(mask) = actual.token_valid() {
            assert_eq!(
                Vec::<u8>::try_from(mask.to_kind(Kind::Uint8).to_device(Device::Cpu).view([-1]))
                    .unwrap(),
                *expected.token_valid.as_ref().unwrap(),
                "{id} mask"
            );
            assert_eq!(
                Vec::<f32>::try_from(
                    actual
                        .incidence()
                        .unwrap()
                        .to_device(Device::Cpu)
                        .view([-1])
                )
                .unwrap(),
                *expected.incidence.as_ref().unwrap(),
                "{id} incidence"
            );
        }
    }
}

#[test]
#[ignore = "requires the pinned full GL4 provider contexts"]
fn finite_packet_budgets_match_source_before_provider_dispatch() {
    crate::torch_device::require_cuda_device(0).unwrap();
    let path = PathBuf::from(std::env::var_os("CNOT103_BUDGET_RUN").unwrap());
    let m = parse_run_manifest(&fs::read(path).unwrap()).unwrap();
    let contexts = m.feature_contexts.build().unwrap();
    let registry = FeatureRegistry::with_all_model_input_providers(contexts.clone()).unwrap();
    let a = &contexts.alphabet;
    assert_eq!(a.dimension(), 4);
    let mut left = MatrixState::identity(4, a.id()).unwrap();
    let mut right = left.clone();
    right
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    let identity = left.clone();
    left.apply_cnot(CnotGate {
        control: 1,
        target: 0,
    })
    .unwrap();
    left.apply_cnot(CnotGate {
        control: 2,
        target: 3,
    })
    .unwrap();
    let states = [left, identity, right];
    let raw: Vec<_> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&w| w as i64))
        .collect();
    let t = Tensor::from_slice(&raw)
        .view([3, 4, 1])
        .to_device(Device::Cuda(0));
    let packed =
        DeviceStateBatch::from_packed_tensor_for_alphabet(t.shallow_clone(), a, 0).unwrap();
    let unbound = DeviceStateBatch::from_packed_tensor(t, 4, 0).unwrap();
    let mut p = FeaturePolicy::all_model_inputs_augmentation_only("resource-policy-source-parity");
    p.solver_assisted = true;
    p.enabled_ids.reverse();
    let provider_rows = 3 * p
        .enabled_ids
        .iter()
        .filter(|&&id| !crate::structural::is_builtin(id))
        .count();
    let byte_limit = 3 * (4 * 4 * 1024 + 4 * 256);
    let mut checked = 0;
    for (evals, bytes) in [
        (None, Some(0)),
        (None, Some(byte_limit - 1)),
        (None, Some(byte_limit)),
        (Some(0), None),
        (Some(1), None),
        (Some(8), None),
        (Some(40), None),
        (Some(92), None),
        (Some(141), None),
        (Some(provider_rows - 1), None),
        (Some(provider_rows), None),
        (Some(31), Some(byte_limit)),
    ] {
        p.evaluations = evals;
        p.max_bytes = bytes;
        let plan = DeviceFeaturePlan::with_contexts(
            &registry,
            &p,
            4,
            0,
            DeviceFeatureContextInputs::from(&contexts),
        )
        .unwrap();
        let reference = registry.build(&states, &p).unwrap();
        let packet = plan.build(&packed).unwrap();
        compare(&packet, &reference, &p);
        if bytes.is_some_and(|n| n < byte_limit) || evals.is_some_and(|n| n <= 8) {
            // Forbidden Stage26/shared-word/endpoint launches demand a bound
            // alphabet, so eager compute-then-mask cannot pass this check.
            compare(&plan.build(&unbound).unwrap(), &reference, &p);
        }
        checked += p.enabled_ids.len() * states.len();
    }
    println!("finite full provider budgets: {checked} feature/row outcomes with compact tensors and masks, provider_rows={provider_rows} global_bytes={byte_limit}");
}

// Provider estimates can exceed the packet estimate even for a single matrix.
// Exercise both sides of the three independent limits and alphabet precedence.
#[test]
fn finite_provider_bytes_match_source_at_the_boundary() {
    use crate::device_feature_plan::{
        BilinearPlanInputs, PositivePacketPlanInputs, SharedRelationPlanInputs,
    };
    use crate::{
        BilinearContext, CnotAlphabet, FeatureId, OptionalGatePacket, SharedRelationContext,
    };
    let device = 0;
    crate::torch_device::require_cuda_device(device).unwrap();
    let bilinear = [BilinearContext::new(
        4,
        vec![0],
        vec![1],
        16,
        vec![vec![1]],
        vec![vec![1]; 16],
        1,
        vec![vec![0]],
        vec![vec![0]],
    )
    .unwrap()];
    let relation = [SharedRelationContext::new(
        4,
        vec![0, 1, 2, 3],
        vec![0, 1, 2, 3],
        vec![vec![1], vec![2], vec![4], vec![8]],
    )
    .unwrap()];
    let positive = [OptionalGatePacket::new(4, vec![(0, 2)], vec![]).unwrap()];
    let mut registry = FeatureRegistry::new();
    crate::register_bilinear_contexts(&mut registry, bilinear.to_vec(), 16, 4096).unwrap();
    crate::register_shared_relation_contexts(&mut registry, relation.to_vec(), 1024, 4, 4).unwrap();
    crate::register_positive_packet_contexts(&mut registry, positive.to_vec(), 4, 4, 1024, 4)
        .unwrap();
    let inputs = DeviceFeatureContextInputs {
        bilinear: Some(BilinearPlanInputs {
            contexts: &bilinear,
            max_dimension: 16,
            max_matrix_bits: 4096,
        }),
        shared_relation: Some(SharedRelationPlanInputs {
            contexts: &relation,
            max_vectors: 1024,
            max_coordinates: 4,
            max_ports: 4,
        }),
        positive_packet: Some(PositivePacketPlanInputs {
            contexts: &positive,
            max_n: 4,
            max_slots: 4,
            max_monomials: 1024,
            max_factor_entries: 4,
        }),
        ..Default::default()
    };
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let other = CnotAlphabet::new(
        4,
        "alternate-labelled-CNOT",
        alphabet.gates().to_vec(),
        vec![1; 12],
    )
    .unwrap();
    for (ordinal, bound) in [
        (5, bilinear[0].working_bytes_upper_bound().unwrap()),
        (
            6,
            crate::shared_relation::working_bytes_upper_bound(4, 4, 4, 4, 1024).unwrap(),
        ),
        (
            8,
            crate::positive_packet::working_bytes_upper_bound(4, 1, 1024).unwrap(),
        ),
    ] {
        assert!(bound > crate::features::workspace_bytes_per_state(4).unwrap());
        let id = FeatureId::new(b'I', ordinal).unwrap();
        for a in [&alphabet, &other] {
            let state = MatrixState::identity(4, a.id()).unwrap();
            let raw = Tensor::from_slice(&[1i64, 2, 4, 8])
                .view([1, 4, 1])
                .to_device(Device::Cuda(device));
            let packed = DeviceStateBatch::from_packed_tensor_for_alphabet(raw, a, device).unwrap();
            for cap in [bound - 1, bound] {
                let mut policy =
                    FeaturePolicy::all_model_inputs_augmentation_only("provider-bytes-parity");
                policy.enabled_ids = vec![id];
                policy.solver_assisted = true;
                policy.max_bytes = Some(cap);
                let source = registry
                    .build(std::slice::from_ref(&state), &policy)
                    .unwrap();
                let status = if ordinal == 8 && a == &other {
                    FeatureStatus::Unsupported
                } else if cap < bound {
                    FeatureStatus::ResourceLimit
                } else {
                    FeatureStatus::Computed
                };
                assert_eq!(source.feature(id).status, vec![status]);
                let plan = DeviceFeaturePlan::with_contexts(&registry, &policy, 4, device, inputs)
                    .unwrap();
                compare(&plan.build(&packed).unwrap(), &source, &policy);
            }
        }
        println!("{id} exact byte boundary {bound}, both alphabet identities");
    }
}
