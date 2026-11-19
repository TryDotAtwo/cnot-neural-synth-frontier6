//! Production GPU feature -> model boundary. Host computation/readback below
//! belongs only to the independent reference and assertions.
use cnot103_native::device_context::DeviceBlockParityContext;
use cnot103_native::device_feature_plan::DeviceFeaturePlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{
    DeviceFeatureEntry, DeviceFeaturePacket, DeviceFeatureStatus, FusionDistanceField, ModelConfig,
};
use cnot103_native::{
    register_block_parities, BlockContext, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    MatrixState,
};
use std::collections::BTreeMap;
use tch::{nn::VarStore, Device, Kind, Tensor};

fn config() -> ModelConfig {
    ModelConfig {
        wire_dim: 8,
        pair_dim: 4,
        structural_blocks: 1,
        attention_heads: 2,
        feedforward_dim: 16,
        fusion_blocks: 1,
        basis_proposal_hidden_dim: 8,
    }
}

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: (1..=20).map(|i| FeatureId::new(b'A', i).unwrap()).collect(),
        seed: 0,
        schema_id: "gpu-model-packet-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn device_packet(
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
) -> DeviceFeaturePacket {
    let n = states[0].dimension();
    let packed: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|v| *v as i64))
        .collect();
    let device_states = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&packed)
            .view([states.len() as i64, n as i64, n.div_ceil(64) as i64])
            .to_device(Device::Cuda(0)),
        n,
        0,
    )
    .unwrap();
    let family = device_states.family_a().unwrap();
    let entries = registry
        .selected_specs(policy)
        .unwrap()
        .into_iter()
        .map(|spec| {
            let values = family
                .feature(spec.id.ordinal() as usize)
                .unwrap()
                .shallow_clone();
            DeviceFeatureEntry::new(
                spec,
                values,
                Tensor::full(
                    [states.len() as i64],
                    DeviceFeatureStatus::Computed as i64,
                    (Kind::Uint8, Device::Cuda(0)),
                ),
                None,
                None,
            )
        })
        .collect();
    DeviceFeaturePacket::new(
        family.feature(1).unwrap().squeeze_dim(-1),
        Tensor::ones(
            [states.len() as i64, n as i64],
            (Kind::Bool, Device::Cuda(0)),
        ),
        policy.schema_id.clone(),
        policy.id().unwrap(),
        entries,
    )
    .unwrap()
}

fn gradients(vars: &VarStore) -> BTreeMap<String, Tensor> {
    vars.variables()
        .into_iter()
        .filter_map(|(name, value)| {
            let gradient = value.grad();
            gradient.defined().then(|| (name, gradient.copy()))
        })
        .collect()
}

fn assert_close(actual: &Tensor, expected: &Tensor, label: &str) {
    assert_eq!(actual.size(), expected.size(), "shape: {label}");
    assert!(
        actual.allclose(expected, 2e-5, 2e-5, false),
        "value mismatch: {label}"
    );
}

fn copy_entry(entry: &DeviceFeatureEntry) -> DeviceFeatureEntry {
    DeviceFeatureEntry::new(
        entry.spec().clone(),
        entry.values().shallow_clone(),
        entry.status().shallow_clone(),
        entry.token_valid().map(Tensor::shallow_clone),
        entry.incidence().map(Tensor::shallow_clone),
    )
}

fn repack(
    packet: &DeviceFeaturePacket,
    policy: &FeaturePolicy,
    entries: Vec<DeviceFeatureEntry>,
) -> Result<DeviceFeaturePacket, cnot103_native::model::ModelError> {
    DeviceFeaturePacket::new(
        packet.matrix().shallow_clone(),
        packet.valid_nodes().shallow_clone(),
        policy.schema_id.clone(),
        policy.id().unwrap(),
        entries,
    )
}

#[test]
fn cuda_producer_packet_matches_reference_model_outputs_and_gradients() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let registry = FeatureRegistry::new();
    let policy = policy();
    for n in [3u32, 65] {
        let identity = MatrixState::identity(n, "gpu-packet-test").unwrap();
        let mut changed = identity.clone();
        changed
            .apply_cnot(CnotGate {
                control: n - 2,
                target: n - 1,
            })
            .unwrap();
        let states = [identity, changed];
        let reference_packet = registry.build(&states, &policy).unwrap();
        let gpu_packet = device_packet(&states, &registry, &policy);
        let vars = VarStore::new(Device::Cuda(0));
        let model = FusionDistanceField::new_for_policy(
            &vars.root(),
            &registry,
            &policy,
            Device::Cuda(0),
            config(),
        )
        .unwrap();

        let reference = model.forward(&reference_packet).unwrap();
        let reference_loss = reference.value.sum(Kind::Float)
            + reference
                .proposal_logits
                .masked_select(&reference.action_mask)
                .square()
                .mean(Kind::Float)
            + reference
                .qhat
                .masked_select(&reference.action_mask)
                .square()
                .mean(Kind::Float);
        reference_loss.backward();
        let expected_gradients = gradients(&vars);
        for mut value in vars.trainable_variables() {
            value.zero_grad();
        }

        let actual = model.forward_device(&gpu_packet).unwrap();
        assert_eq!(actual.value.device(), Device::Cuda(0));
        assert_eq!(
            actual.value.double_value(&[0]),
            0.0,
            "V(I) must be exactly zero"
        );
        assert_close(&actual.value, &reference.value, "distance");
        assert_eq!(
            Vec::<i64>::try_from(
                &actual
                    .action_mask
                    .to_device(Device::Cpu)
                    .to_kind(Kind::Int64)
                    .view([-1])
            )
            .unwrap(),
            Vec::<i64>::try_from(
                &reference
                    .action_mask
                    .to_device(Device::Cpu)
                    .to_kind(Kind::Int64)
                    .view([-1])
            )
            .unwrap()
        );
        assert_close(
            &actual.proposal_logits.masked_select(&actual.action_mask),
            &reference
                .proposal_logits
                .masked_select(&reference.action_mask),
            "proposal",
        );
        assert_close(
            &actual.qhat.masked_select(&actual.action_mask),
            &reference.qhat.masked_select(&reference.action_mask),
            "qhat",
        );
        let actual_loss = actual.value.sum(Kind::Float)
            + actual
                .proposal_logits
                .masked_select(&actual.action_mask)
                .square()
                .mean(Kind::Float)
            + actual
                .qhat
                .masked_select(&actual.action_mask)
                .square()
                .mean(Kind::Float);
        actual_loss.backward();
        let actual_gradients = gradients(&vars);
        assert_eq!(
            actual_gradients.keys().collect::<Vec<_>>(),
            expected_gradients.keys().collect::<Vec<_>>()
        );
        assert!(!actual_gradients.is_empty());
        for (name, expected) in expected_gradients {
            assert_close(&actual_gradients[&name], &expected, &name);
        }
    }
}

#[test]
fn cuda_missing_reasons_are_distinct_from_computed_zero_and_train_their_own_embedding() {
    use DeviceFeatureStatus::*;
    let device = Device::Cuda(0);
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let mut policy = policy();
    policy.enabled_ids.truncate(2);
    policy.schema_id = cnot103_native::schema::MISSING_TOKEN_LAYOUT_ID.to_owned();
    let registry = FeatureRegistry::new();
    let vars = VarStore::new(device);
    let model =
        FusionDistanceField::new_for_policy(&vars.root(), &registry, &policy, device, config())
            .unwrap();
    let mut state = MatrixState::identity(3, "gpu-packet-test").unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let packet = device_packet(&[state], &registry, &policy);
    let a01 = packet.feature(policy.enabled_ids[0]).unwrap();
    let a02 = packet.feature(policy.enabled_ids[1]).unwrap();
    let embedding = vars
        .variables()
        .remove("stem.missing_status.weight")
        .unwrap();
    let mut predictions = Vec::new();
    for (i, status) in [Computed, Inapplicable, Unsupported, ResourceLimit]
        .into_iter()
        .enumerate()
    {
        for mut variable in vars.trainable_variables() {
            variable.zero_grad();
        }
        let predict = |payload: f64| {
            let entry = DeviceFeatureEntry::new(
                a02.spec().clone(),
                a02.values().full_like(payload),
                Tensor::full([1], status as i64, (Kind::Uint8, device)),
                None,
                None,
            );
            model
                .forward_device(&repack(&packet, &policy, vec![copy_entry(a01), entry]).unwrap())
                .unwrap()
                .value
        };
        let value = predict(if i > 0 { f64::NAN } else { 0.0 });
        let y = value.double_value(&[0]);
        if i > 0 {
            assert_eq!(y, predict(0.0).double_value(&[0]));
            assert_eq!(
                y,
                predict(123.0).double_value(&[0]),
                "masked payload reached the model"
            );
        }
        value.sum(Kind::Float).backward();
        let grad = embedding
            .grad()
            .abs()
            .sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        let columns = Vec::<f32>::try_from(grad.to_device(Device::Cpu)).unwrap();
        for (column, magnitude) in columns.into_iter().enumerate() {
            assert_eq!(
                magnitude > 0.0,
                i > 0 && column == 2 + i,
                "gradient for {status:?}, column {column}"
            );
        }
        assert!(
            predictions.iter().all(|p: &f64| (p - y).abs() > 1e-8),
            "missing reason collided with another output"
        );
        predictions.push(y);
    }
}

#[test]
fn cuda_packet_rejects_cpu_payloads_missing_inputs_and_wrong_contracts() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    assert!(DeviceFeaturePacket::new(
        Tensor::eye(3, (Kind::Float, Device::Cpu)).unsqueeze(0),
        Tensor::ones([1, 3], (Kind::Bool, Device::Cpu)),
        "schema",
        "policy",
        vec![],
    )
    .is_err());
    let registry = FeatureRegistry::new();
    let full_policy = policy();
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &registry,
        &full_policy,
        Device::Cuda(0),
        config(),
    )
    .unwrap();
    let states = [MatrixState::identity(3, "gpu-packet-test").unwrap()];
    let mut subset_policy = full_policy.clone();
    subset_policy.enabled_ids.pop();
    let subset = device_packet(&states, &registry, &subset_policy);
    // Preserve the full policy identity so this tests missing entries, rather
    // than merely exercising the policy-identity rejection.
    let subset = repack(
        &subset,
        &full_policy,
        subset_policy
            .enabled_ids
            .iter()
            .map(|id| copy_entry(subset.feature(*id).unwrap()))
            .collect(),
    )
    .unwrap();
    assert!(
        model.forward_device(&subset).is_err(),
        "missing selected feature must not be silently zero-filled"
    );
    let mut other_policy = full_policy;
    other_policy.schema_id = "wrong-schema".into();
    assert!(model
        .forward_device(&device_packet(&states, &registry, &other_policy))
        .is_err());
}

#[test]
fn cuda_packet_rejects_malformed_shapes_duplicates_and_diagnostic_inputs() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let registry = FeatureRegistry::new();
    let policy = policy();
    let packet = device_packet(
        &[MatrixState::identity(3, "gpu-packet-test").unwrap()],
        &registry,
        &policy,
    );
    let id = FeatureId::new(b'A', 1).unwrap();
    let entry = packet.feature(id).unwrap();
    assert!(repack(&packet, &policy, vec![copy_entry(entry), copy_entry(entry)]).is_err());
    assert!(repack(
        &packet,
        &policy,
        vec![DeviceFeatureEntry::new(
            entry.spec().clone(),
            Tensor::zeros([1, 3, 2, 1], (Kind::Float, Device::Cuda(0))),
            entry.status().shallow_clone(),
            None,
            None
        )]
    )
    .is_err());
    assert!(repack(
        &packet,
        &policy,
        vec![DeviceFeatureEntry::new(
            entry.spec().clone(),
            entry.values().to_device(Device::Cpu),
            entry.status().shallow_clone(),
            None,
            None
        )]
    )
    .is_err());
    let mut h07 = entry.spec().clone();
    h07.id = FeatureId::new(b'H', 7).unwrap();
    assert!(repack(
        &packet,
        &policy,
        vec![DeviceFeatureEntry::new(
            h07,
            entry.values().shallow_clone(),
            entry.status().shallow_clone(),
            None,
            None
        )]
    )
    .is_err());
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &registry,
        &policy,
        Device::Cuda(0),
        config(),
    )
    .unwrap();
    let entries = policy
        .enabled_ids
        .iter()
        .map(|id| {
            let entry = packet.feature(*id).unwrap();
            let mut spec = entry.spec().clone();
            if *id == FeatureId::new(b'A', 1).unwrap() {
                spec.producer_version.push_str("-different");
            }
            DeviceFeatureEntry::new(
                spec,
                entry.values().shallow_clone(),
                entry.status().shallow_clone(),
                None,
                None,
            )
        })
        .collect();
    assert!(model
        .forward_device(&repack(&packet, &policy, entries).unwrap())
        .is_err());
}

#[test]
fn cuda_all_feature_roles_and_resource_masks_match_reference() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let device = Device::Cuda(0);
    let a01 = FeatureId::new(b'A', 1).unwrap();
    let d05 = FeatureId::new(b'D', 5).unwrap();
    let d08 = FeatureId::new(b'D', 8).unwrap();
    let f01 = FeatureId::new(b'F', 1).unwrap();
    let identity = MatrixState::identity(4, "gpu-block-model-test").unwrap();
    let mut changed = identity.clone();
    for (control, target) in [(0, 2), (1, 2), (1, 3)] {
        changed.apply_cnot(CnotGate { control, target }).unwrap();
    }
    let states = [changed, identity];
    let packed: Vec<i64> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|v| *v as i64))
        .collect();
    let gpu_states = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&packed)
            .view([2, 4, 1])
            .to_device(device),
        4,
        0,
    )
    .unwrap();
    let contexts = [BlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap()];
    for budget in [2, 8] {
        let mut registry = FeatureRegistry::new();
        register_block_parities(&mut registry, contexts.to_vec(), budget, 64).unwrap();
        let mut policy = policy();
        policy.enabled_ids = vec![a01, d05, d08, f01];
        policy.symmetry_contract = "augmentation_only".into();
        let reference_packet = registry.build(&states, &policy).unwrap();
        let context = DeviceBlockParityContext::new(4, &contexts, 0, budget).unwrap();
        let family = gpu_states.family_a().unwrap();
        let slice = gpu_states.feature_slice().unwrap();
        let node = gpu_states.block_parity_d05(&context).unwrap();
        let group = gpu_states.block_parity_d08(&context).unwrap();
        let computed = Tensor::full(
            [2],
            DeviceFeatureStatus::Computed as i64,
            (Kind::Uint8, device),
        );
        let limited = Tensor::full(
            [2],
            DeviceFeatureStatus::ResourceLimit as i64,
            (Kind::Uint8, device),
        );
        let node_status = computed.where_self(&node.status().squeeze_dim(1).eq(1), &limited);
        let group_status = computed.where_self(&group.status().squeeze_dim(1).eq(1), &limited);
        // Independent expected status: transformed block has two nonzero
        // patterns plus zero; identity has only zero.
        assert_eq!(
            Vec::<u8>::try_from(&group_status.to_device(Device::Cpu)).unwrap(),
            if budget == 2 { vec![4, 1] } else { vec![1, 1] }
        );
        let specs = registry.specs();
        let entries = vec![
            DeviceFeatureEntry::new(
                specs[a01.index()].clone(),
                family.feature(1).unwrap().shallow_clone(),
                computed.shallow_clone(),
                None,
                None,
            ),
            DeviceFeatureEntry::new(
                specs[d05.index()].clone(),
                node.feature(d05).unwrap().squeeze_dim(1),
                node_status,
                None,
                None,
            ),
            DeviceFeatureEntry::new(
                specs[d08.index()].clone(),
                group.feature(d08).unwrap().shallow_clone(),
                group_status,
                Some(group.token_valid().shallow_clone()),
                Some(group.incidence().shallow_clone()),
            ),
            DeviceFeatureEntry::new(
                specs[f01.index()].clone(),
                slice.feature(f01).unwrap().shallow_clone(),
                computed,
                None,
                None,
            ),
        ];
        let packet = DeviceFeaturePacket::new(
            family.feature(1).unwrap().squeeze_dim(-1),
            Tensor::ones([2, 4], (Kind::Bool, device)),
            policy.schema_id.clone(),
            policy.id().unwrap(),
            entries,
        )
        .unwrap();
        let vars = VarStore::new(device);
        let model =
            FusionDistanceField::new_for_policy(&vars.root(), &registry, &policy, device, config())
                .unwrap();
        let expected = model.forward(&reference_packet).unwrap();
        let actual = model.forward_device(&packet).unwrap();
        assert_close(&actual.value, &expected.value, "all-role distance");
        assert_close(
            &actual.proposal_logits.masked_select(&actual.action_mask),
            &expected
                .proposal_logits
                .masked_select(&expected.action_mask),
            "all-role proposal",
        );
        assert_close(
            &actual.qhat.masked_select(&actual.action_mask),
            &expected.qhat.masked_select(&expected.action_mask),
            "all-role qhat",
        );
        actual.value.sum(Kind::Float).backward();
        for gradient in gradients(&vars).values() {
            assert!(gradient.isfinite().all().int64_value(&[]) != 0);
        }
    }
}

#[test]
fn cuda_compiled_feature_plan_feeds_model_and_refuses_missing_producers() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let mut registry = FeatureRegistry::new();
    cnot103_native::register_symmetry_alignment_h06(&mut registry).unwrap();
    let mut selected = policy();
    selected
        .enabled_ids
        .extend((1..=12).map(|i| FeatureId::new(b'B', i).unwrap()));
    selected.enabled_ids.extend(
        [
            (b'C', 1),
            (b'D', 3),
            (b'F', 1),
            (b'G', 5),
            (b'H', 1),
            (b'H', 2),
            (b'H', 3),
            (b'H', 4),
            (b'H', 5),
            (b'H', 6),
        ]
        .into_iter()
        .map(|(family, ordinal)| FeatureId::new(family, ordinal).unwrap()),
    );
    assert_eq!(selected.enabled_ids.len(), 42);
    let plan = DeviceFeaturePlan::context_free(&registry, &selected, 0).unwrap();
    let mut state = MatrixState::identity(3, "compiled-device-plan").unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    let packed: Vec<i64> = state.packed_rows().iter().map(|v| *v as i64).collect();
    let states = DeviceStateBatch::from_packed_tensor(
        Tensor::from_slice(&packed)
            .view([1, 3, 1])
            .to_device(Device::Cuda(0)),
        3,
        0,
    )
    .unwrap();
    let packet = plan.build(&states).unwrap();
    let reference = registry.build(&[state], &selected).unwrap();
    for id in &selected.enabled_ids {
        let actual = packet.feature(*id).unwrap();
        assert_eq!(actual.values().device(), Device::Cuda(0));
        assert_eq!(actual.spec(), &reference.feature(*id).spec);
        assert_eq!(
            Vec::<f32>::try_from(
                &actual
                    .values()
                    .to_device(Device::Cpu)
                    .contiguous()
                    .view([-1])
            )
            .unwrap(),
            reference.feature(*id).values,
            "{id}"
        );
    }
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &registry,
        &selected,
        Device::Cuda(0),
        config(),
    )
    .unwrap();
    let expected = model.forward(&reference).unwrap();
    let actual = model.forward_device(&packet).unwrap();
    assert_close(&actual.value, &expected.value, "42-feature compiled plan");
    assert_close(
        &actual.qhat.masked_select(&actual.action_mask),
        &expected.qhat.masked_select(&expected.action_mask),
        "42-feature qhat",
    );
    let full = FeaturePolicy::all_model_inputs_augmentation_only("full-runtime-must-refuse");
    let missing = DeviceFeaturePlan::context_free(&registry, &full, 0)
        .err()
        .expect("an incomplete producer set must not be admitted");
    assert!(missing.contains("C02") && missing.contains("I11"));
    let mut bounded = selected;
    bounded.max_bytes = Some(1);
    let limited = DeviceFeaturePlan::context_free(&registry, &bounded, 0)
        .unwrap()
        .build(&states)
        .unwrap();
    assert!(limited.matrix().equal(packet.matrix()));
    for id in &bounded.enabled_ids {
        let entry = limited.feature(*id).unwrap();
        assert_eq!(entry.status().int64_value(&[0]), 4, "{id} byte budget");
        assert_eq!(entry.values().count_nonzero(None).int64_value(&[]), 0);
    }
}
