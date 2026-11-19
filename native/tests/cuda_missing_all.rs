//! Every optional model input carries a separate learned token per missing reason.
//! Payloads here are synthetic; feature shapes and roles come from the real registry.
use cnot103_native::model::{
    DeviceFeatureEntry, DeviceFeaturePacket, DeviceFeatureStatus, FusionDistanceField, ModelConfig,
};
use cnot103_native::schema::{FeatureRole, FeatureSpec, MISSING_TOKEN_LAYOUT_ID, MODEL_INPUT_COUNT};
use cnot103_native::{FeatureId, FeaturePolicy, FeatureRegistry};
use tch::{nn::VarStore, Device, Kind, Tensor};

#[path = "support/feature_contexts.rs"]
mod feature_contexts;

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

fn packet(specs: &[FeatureSpec], policy: &FeaturePolicy, payload: f64) -> DeviceFeaturePacket {
    let device = Device::Cuda(0);
    let batch = 1 + (specs.len() - 1) * 3;
    let n = 3i64;
    let batch_i64 = batch as i64;
    let matrix = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 0., 0., 0., 1.])
        .view([1, n, n])
        .to_device(device)
        .expand([batch_i64, n, n], true);
    let valid = Tensor::ones([batch_i64, n], (Kind::Bool, device));
    let entries = specs
        .iter()
        .enumerate()
        .map(|(feature_index, spec)| {
            let mut statuses = vec![DeviceFeatureStatus::Computed as u8; batch];
            if feature_index > 0 {
                let row = 1 + (feature_index - 1) * 3;
                statuses[row] = DeviceFeatureStatus::Inapplicable as u8;
                statuses[row + 1] = DeviceFeatureStatus::Unsupported as u8;
                statuses[row + 2] = DeviceFeatureStatus::ResourceLimit as u8;
            }
            let status = Tensor::from_slice(&statuses).to_device(device);
            let missing = status.ne(DeviceFeatureStatus::Computed as i64);
            let channels = spec.channels as i64;
            let (shape, token_valid, incidence) = match spec.role {
                FeatureRole::Node => (vec![batch_i64, n, channels], None, None),
                FeatureRole::Pair => (vec![batch_i64, n, n, channels], None, None),
                FeatureRole::Global => (vec![batch_i64, channels], None, None),
                FeatureRole::Group => {
                    let incidence = Tensor::ones(
                        [batch_i64, 1, n, spec.wire_roles.len() as i64],
                        (Kind::Float, device),
                    )
                    .masked_fill(&missing.view([batch_i64, 1, 1, 1]), payload);
                    (
                        vec![batch_i64, 1, channels],
                        Some(Tensor::ones([batch_i64, 1], (Kind::Bool, device))),
                        Some(incidence),
                    )
                }
            };
            let mut broadcast = vec![1; shape.len()];
            broadcast[0] = batch_i64;
            let values = Tensor::zeros(shape.as_slice(), (Kind::Float, device))
                .masked_fill(&missing.view(broadcast.as_slice()), payload);
            DeviceFeatureEntry::new(spec.clone(), values, status, token_valid, incidence)
        })
        .collect();
    DeviceFeaturePacket::new(matrix, valid, policy.schema_id.clone(), policy.id().unwrap(), entries)
        .unwrap()
}

#[test]
fn all_optional_features_mask_payload_and_train_each_reason_token() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let registry = FeatureRegistry::with_all_model_input_providers(
        feature_contexts::complete_model_input_contexts(),
    )
    .unwrap();
    let mut policy = FeaturePolicy::all_model_inputs_augmentation_only(MISSING_TOKEN_LAYOUT_ID);
    policy.solver_assisted = true;
    let specs = registry.selected_specs(&policy).unwrap();
    assert_eq!(specs.len(), MODEL_INPUT_COUNT);
    assert_eq!(specs[0].id, FeatureId::new(b'A', 1).unwrap());

    tch::manual_seed(1497);
    let vars = VarStore::new(Device::Cuda(0));
    let model = FusionDistanceField::new_for_policy(
        &vars.root(), &registry, &policy, Device::Cuda(0), config(),
    )
    .unwrap();
    let run = |payload| model.forward_device(&packet(&specs, &policy, payload)).unwrap();
    let missing_nan = run(f64::NAN);
    let missing_zero = run(0.0);
    let missing_large = run(123.0);
    let mask = &missing_nan.action_mask;
    let compared = [
        ("value/zero", missing_nan.value.shallow_clone(), missing_zero.value.shallow_clone()),
        ("value/large", missing_nan.value.shallow_clone(), missing_large.value.shallow_clone()),
        ("qhat/zero", missing_nan.qhat.masked_select(mask), missing_zero.qhat.masked_select(mask)),
        ("qhat/large", missing_nan.qhat.masked_select(mask), missing_large.qhat.masked_select(mask)),
        ("proposal/zero", missing_nan.proposal_logits.masked_select(mask), missing_zero.proposal_logits.masked_select(mask)),
        ("proposal/large", missing_nan.proposal_logits.masked_select(mask), missing_large.proposal_logits.masked_select(mask)),
    ];
    for (name, actual, expected) in compared {
        assert!(actual.isfinite().all().int64_value(&[]) != 0, "nonfinite {name}");
        assert!(actual.allclose(&expected, 1e-6, 1e-6, false), "payload leaked: {name}");
    }

    missing_nan.value.sum(Kind::Float).backward();
    let weight = vars.variables().remove("stem.missing_status.weight").unwrap();
    let magnitude = weight.grad().abs().sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
    let columns = Vec::<f32>::try_from(magnitude.to_device(Device::Cpu)).unwrap();
    assert_eq!(columns.len(), MODEL_INPUT_COUNT * 3);
    assert!(columns[..3].iter().all(|value| *value == 0.0));
    for (index, magnitude) in columns.iter().copied().enumerate().skip(3) {
        let spec = &specs[index / 3];
        assert!(magnitude.is_finite() && magnitude > 0.0,
                "no embedding gradient for {} reason {}", spec.id, index % 3);
    }
}
