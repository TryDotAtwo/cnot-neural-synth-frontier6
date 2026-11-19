use cnot103_native::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan};
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::{
    register_diffusion, DiffusionConfig, FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus,
    MatrixState,
};
use serde_json::Value;
use std::collections::BTreeSet;
use tch::{Device, Tensor};

#[path = "cuda_column_diffusion/pair.rs"]
mod pair;
#[path = "cuda_column_diffusion/shared_word.rs"]
mod shared_word;

fn policy(ordinals: &[u8]) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: ordinals
            .iter()
            .map(|&i| FeatureId::new(b'G', i).unwrap())
            .collect(),
        seed: 91,
        schema_id: "column-diffusion-parity".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn registry(config: DiffusionConfig) -> FeatureRegistry {
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, config).unwrap();
    registry
}

fn bind(
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    n: u32,
    config: DiffusionConfig,
) -> Result<DeviceFeaturePlan, String> {
    DeviceFeaturePlan::with_contexts(
        registry,
        policy,
        n,
        0,
        DeviceFeatureContextInputs {
            diffusion: Some(config),
            ..Default::default()
        },
    )
}

fn device_states(states: &[MatrixState]) -> DeviceStateBatch {
    let n = states[0].dimension();
    let packed: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&w| w as i64))
        .collect();
    let tensor = Tensor::from_slice(&packed)
        .view([states.len() as i64, n as i64, n.div_ceil(64) as i64])
        .to_device(Device::Cuda(0));
    DeviceStateBatch::from_packed_tensor(tensor, n, 0).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    Vec::try_from(tensor.to_device(Device::Cpu).view([-1])).unwrap()
}

#[test]
fn configured_device_owns_profile_and_packet_on_two_visible_gpus() {
    fn current_device() -> i32 {
        unsafe extern "C" {
            fn cudaGetDevice(index: &mut i32) -> i32;
        }
        let mut device = -1;
        assert_eq!(unsafe { cudaGetDevice(&mut device) }, 0);
        device
    }
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    if tch::Cuda::device_count() < 2 {
        return;
    }
    let config = DiffusionConfig::default();
    let registry = registry(config);
    let selected = shared_word::selection(&[1, 2, 3, 4, 6, 7, 8, 9, 10], 91);
    let states = class_cover(4);
    let reference = registry.build(&states, &selected).unwrap();
    let packed = device_states(&states);
    let original_device = current_device();
    for index in [1, 0] {
        let batch = DeviceStateBatch::from_packed_tensor(
            packed.tensor().to_device(Device::Cuda(index)),
            4,
            index,
        )
        .unwrap();
        let plan = DeviceFeaturePlan::with_contexts(
            &registry,
            &selected,
            4,
            index,
            DeviceFeatureContextInputs {
                diffusion: Some(config),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(current_device(), original_device);
        let result = plan.build(&batch).unwrap();
        assert_eq!(current_device(), original_device);
        for &id in &selected.enabled_ids {
            let actual = result.feature(id).unwrap().values();
            assert_eq!(actual.device(), Device::Cuda(index));
            assert_eq!(values(actual), reference.feature(id).values);
        }
    }
}

fn compare(
    registry: &FeatureRegistry,
    plan: &DeviceFeaturePlan,
    policy: &FeaturePolicy,
    states: &[MatrixState],
) {
    let source = registry.build(states, policy).unwrap();
    let packet = plan.build(&device_states(states)).unwrap();
    for &id in &policy.enabled_ids {
        let actual = packet.feature(id).unwrap();
        let expected = source.feature(id);
        assert_eq!(actual.spec(), &expected.spec);
        assert_eq!(actual.values().device(), Device::Cuda(0));
        let codes: Vec<u8> = expected
            .status
            .iter()
            .map(|s| match s {
                FeatureStatus::Computed => 1,
                FeatureStatus::ResourceLimit => 4,
                _ => panic!("unexpected source {id} status {s:?}"),
            })
            .collect();
        assert_eq!(
            Vec::<u8>::try_from(actual.status().to_device(Device::Cpu)).unwrap(),
            codes
        );
        let gpu = values(actual.values());
        assert_eq!(gpu.len(), expected.values.len());
        for (index, (&value, &reference)) in gpu.iter().zip(&expected.values).enumerate() {
            assert_eq!(
                value,
                reference,
                "{id} n={} element {index}",
                states[0].dimension()
            );
        }
    }
}

#[test]
fn pinned_gl2_gl3_and_sampled_gl4_to_gl6_match_all_source_channels() {
    let config = DiffusionConfig {
        taus: [0., 0.25, 1., 4.],
        max_n: 16,
        mc_samples: 128,
        ..Default::default()
    };
    let registry = registry(config);
    let selected = policy(&[1, 2, 3, 4, 6]);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/python_diffusion_10.json")).unwrap();
    for batch in fixture["batches"].as_array().unwrap() {
        let n = batch["n"].as_u64().unwrap() as u32;
        let states: Vec<_> = batch["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let rows = s["packed_rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|word| word.as_u64().unwrap())
                    .collect();
                MatrixState::from_packed_rows(n, "column-diffusion-test", rows).unwrap()
            })
            .collect();
        assert_eq!(
            states.len(),
            match n {
                2 => 6,
                3 => 168,
                _ => 12,
            }
        );
        let plan = bind(&registry, &selected, n, config).unwrap();
        compare(&registry, &plan, &selected, &states);
        let sampled = shared_word::selection(&[7, 8, 9, 10], 91);
        let sampled_plan = bind(&registry, &sampled, n, config).unwrap();
        compare(&registry, &sampled_plan, &sampled, &states);
    }
}

// Lower triangular matrices cover every diagonal-one class. A cyclic row
// shift covers the other n-1 classes. Transposes exchange the two projections.
fn class_cover(n: u32) -> Vec<MatrixState> {
    let n = n as usize;
    let words = n.div_ceil(64);
    let mut result = Vec::new();
    for shifted in [false, true] {
        for transpose in [false, true] {
            let mut packed = vec![0u64; n * words];
            for i in 0..n {
                let old_row = if shifted { (i + n - 1) % n } else { i };
                for j in 0..=old_row {
                    let (r, c) = if transpose { (j, i) } else { (i, j) };
                    packed[r * words + c / 64] |= 1u64 << (c % 64);
                }
            }
            result.push(
                MatrixState::from_packed_rows(n as u32, "column-diffusion-test", packed).unwrap(),
            );
        }
    }
    for rows in [false, true] {
        let mut seen = BTreeSet::new();
        for s in &result {
            let bit = |r: usize, c: usize| (s.packed_rows()[r * words + c / 64] >> (c % 64)) & 1;
            for i in 0..n {
                let diagonal = bit(i, i);
                let weight: u64 = (0..n)
                    .map(|j| if rows { bit(i, j) } else { bit(j, i) })
                    .sum();
                seen.insert((diagonal, weight - diagonal));
            }
        }
        assert_eq!(seen.len(), 2 * n - 1);
    }
    result
}

#[test]
fn every_class_at_wide_widths_includes_zero_underflow_and_floor_flags() {
    for n in [65, 144] {
        let config = DiffusionConfig {
            taus: [0., 1e-14, 0.75, 16.],
            floor: 1e-12,
            ..Default::default()
        };
        let registry = registry(config);
        let selected = policy(&[1, 2, 3, 4]);
        let plan = bind(&registry, &selected, n, config).unwrap();
        let states = class_cover(n);
        compare(&registry, &plan, &selected, &states);
        // Packet consumers may write to an output; cached descriptors are private.
        let packet = plan.build(&device_states(&states[1..2])).unwrap();
        let entry = packet.feature(FeatureId::new(b'G', 3).unwrap()).unwrap();
        let _ = entry.values().shallow_clone().fill_(42.);
        let _ = entry.status().shallow_clone().fill_(4);
        compare(&registry, &plan, &selected, &states[1..2]);
    }
}

#[test]
fn zero_floor_only_rejects_nonfinite_selected_values_and_preserves_signed_deltas() {
    let config = DiffusionConfig {
        taus: [0., 0.2, 3., 0.],
        floor: 0.,
        ..Default::default()
    };
    let registry = registry(config);
    let identity = MatrixState::identity(4, "column-diffusion-test").unwrap();
    let changed = class_cover(4);
    for mask in 1..16 {
        let ordinals: Vec<_> = (1..=4).filter(|i| mask & (1 << (i - 1)) != 0).collect();
        let selected = policy(&ordinals);
        let plan = bind(&registry, &selected, 4, config).unwrap();
        compare(&registry, &plan, &selected, std::slice::from_ref(&identity));
        if ordinals == [4] {
            compare(&registry, &plan, &selected, &changed);
        } else {
            assert!(registry
                .build(&changed, &selected)
                .unwrap_err()
                .0
                .contains("non-finite"));
        }
    }
}

#[test]
fn profile_failures_and_configuration_identity_follow_the_selected_producer() {
    let selected = policy(&[1, 2, 3, 4]);
    let states = class_cover(4);
    for config in [
        DiffusionConfig {
            max_n: 3,
            ..Default::default()
        },
        DiffusionConfig {
            taus: [0., 1., 4., 250_000.],
            ..Default::default()
        },
        DiffusionConfig {
            taus: [f64::MAX, 1., 4., 0.],
            ..Default::default()
        },
    ] {
        let registry = registry(config);
        let plan = bind(&registry, &selected, 4, config).unwrap();
        compare(&registry, &plan, &selected, &states);
    }
    let config = DiffusionConfig::default();
    let registry = registry(config);
    for mismatch in [
        DiffusionConfig {
            floor: 0.1,
            ..config
        },
        DiffusionConfig {
            taus: [0.; 4],
            ..config
        },
        DiffusionConfig {
            max_n: 17,
            ..config
        },
        DiffusionConfig {
            max_pair_states: 7,
            ..config
        },
    ] {
        assert!(bind(&registry, &selected, 4, mismatch)
            .err()
            .unwrap()
            .contains("metadata"));
    }
    let mc_only_change = DiffusionConfig {
        mc_samples: 17,
        mc_tau: 3.,
        ..config
    };
    assert!(bind(&registry, &selected, 4, mc_only_change).is_ok());
    assert!(
        DeviceFeaturePlan::with_contexts(&registry, &selected, 4, 0, Default::default()).is_err()
    );
    assert!(bind(&registry, &policy(&[7]), 4, config).is_err());
    assert!(bind(
        &registry,
        &selected,
        4,
        DiffusionConfig {
            floor: f64::NAN,
            ..config
        }
    )
    .is_err());
}
