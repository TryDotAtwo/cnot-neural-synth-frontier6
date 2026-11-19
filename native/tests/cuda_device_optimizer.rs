use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::nn_training::{AdamWConfig, DeviceUpdateLoss, NeuralTrainer};
use cnot103_native::{
    register_labeled_cuts, FeatureId, FeaturePolicy, FeatureRegistry, LabeledCut, Sampler,
};
use std::collections::BTreeMap;
use tch::{nn::VarStore, Device, Kind, Tensor};

fn fixture(device: Device) -> (VarStore, FusionDistanceField, NeuralTrainer) {
    let vars = VarStore::new(device);
    let mut registry = FeatureRegistry::new();
    register_labeled_cuts(&mut registry, vec![LabeledCut::new(3, vec![0]).unwrap()]).unwrap();
    let mut p = FeaturePolicy::all_model_inputs_augmentation_only("device-adamw");
    p.enabled_ids = vec![
        FeatureId::new(b'A', 1).unwrap(),
        FeatureId::new(b'C', 2).unwrap(),
    ];
    let model = FusionDistanceField::new_for_policy(
        &vars.root(),
        &registry,
        &p,
        device,
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
    let cfg = AdamWConfig {
        learning_rate: 0.003,
        beta1: 0.7,
        beta2: 0.86,
        weight_decay: 0.23,
        epsilon: 1e-8,
    };
    let trainer = NeuralTrainer::new(&vars, Sampler::new(3, 49).unwrap(), cfg).unwrap();
    (vars, model, trainer)
}

fn loss(vars: &VarStore, step: usize, padded: bool) -> Tensor {
    let params = vars.variables();
    let base = &params["stem.raw_matrix.weight"];
    let group = &params["stem.group_roles.C02.0.weight"];
    let mut value = base.square().mean(Kind::Float);
    if matches!(step, 0 | 3) {
        value += group.square().mean(Kind::Float);
    } else if padded {
        value += group.sum(Kind::Float) * 0.0;
    }
    if step == 2 {
        // Finite value and infinite/NaN gradient. This attempt consumes the sample only.
        value += (base - base.detach()).abs().sqrt().sum(Kind::Float);
    }
    value
}
#[path = "support/optimizer_snapshot.rs"]
mod optimizer_snapshot;
use optimizer_snapshot::saved;
#[test]
fn resident_adamw_skips_absent_groups_decay_and_moments_and_rejects_bad_gradients() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let device = Device::Cuda(0);
    let (reference, _model, mut reference_trainer) = fixture(device);
    let (mut vars, model, mut trainer) = fixture(device);
    vars.copy(&reference).unwrap();
    let mut ref_rows = Vec::new();
    for i in 0..5 {
        let outcome = reference_trainer
            .step(2, |indices| {
                ref_rows.extend_from_slice(indices);
                Ok(loss(&reference, i, false))
            })
            .unwrap();
        assert_eq!(outcome.accepted, i != 2);
    }
    let mut i = 0usize;
    let mut rows = Vec::new();
    let outcome = trainer
        .run_device_updates(&model, 5, 2, |indices| {
            rows.extend_from_slice(indices);
            let active = matches!(i, 0 | 3);
            let batch = DeviceUpdateLoss {
                loss: loss(&vars, i, true),
                valid: Tensor::ones([], (Kind::Bool, device)),
                group_activity: Tensor::full([1], i64::from(active), (Kind::Bool, device)),
            };
            i += 1;
            Ok(batch)
        })
        .unwrap();
    assert!(outcome.accepted);
    assert_eq!(rows, ref_rows);
    assert_eq!(trainer.progress(), reference_trainer.progress());
    assert_eq!(trainer.progress().accepted, 4);
    assert_eq!(
        trainer.sampler_snapshot(),
        reference_trainer.sampler_snapshot()
    );
    let reference_params = reference.variables();
    for (name, param) in vars.variables() {
        assert!(
            param.allclose(&reference_params[&name], 2e-6, 2e-7, false),
            "{name}"
        );
    }
    let (mh, m) = saved(&vars, &trainer, "device");
    let (rh, r) = saved(&reference, &reference_trainer, "reference");
    assert_eq!(mh, rh);
    assert_eq!(m.len(), r.len());
    assert!(m.iter().any(|(_, t)| t.abs().max().double_value(&[]) > 0.0));
    for (name, t) in m {
        assert!(t.allclose(&r[&name], 3e-6, 2e-8, false), "{name}");
    }
}

#[test]
fn invalid_objective_vote_rolls_back_the_interval_and_permits_the_same_retry() {
    let index = usize::from(tch::Cuda::device_count() > 1);
    cnot103_native::torch_device::require_cuda_device(index).unwrap();
    let device = Device::Cuda(index);
    let (vars, model, mut trainer) = fixture(device);
    let initial: BTreeMap<_, _> = vars
        .variables()
        .into_iter()
        .map(|(name, t)| (name, t.detach().copy()))
        .collect();
    let sampler = trainer.sampler_snapshot();
    let (metadata, moments) = saved(&vars, &trainer, "vote-before");
    let compute = |valid| DeviceUpdateLoss {
        loss: loss(&vars, 0, false),
        valid: Tensor::full([], i64::from(valid), (Kind::Bool, device)),
        group_activity: Tensor::ones([1], (Kind::Bool, device)),
    };
    let mut attempted_rows = Vec::new();
    let error = trainer
        .run_device_updates(&model, 2, 1, |indices| {
            let valid = attempted_rows.is_empty();
            if !valid {
                let raw = "stem.raw_matrix.weight";
                assert!(!vars.variables()[raw].equal(&initial[raw]));
            }
            attempted_rows.extend_from_slice(indices);
            Ok(compute(valid))
        })
        .unwrap_err();
    assert!(error.to_string().contains("device objective"), "{error}");
    assert_eq!(trainer.progress().attempted, 0);
    assert_eq!(trainer.sampler_snapshot(), sampler);
    for (name, param) in vars.variables() {
        assert!(param.equal(&initial[&name]), "{name}");
    }
    let (after, tensors) = saved(&vars, &trainer, "vote-rollback");
    assert_eq!(metadata, after);
    assert_eq!(moments.len(), tensors.len());
    for (name, moment) in moments {
        assert!(moment.equal(&tensors[&name]), "{name}");
    }
    let mut retry_rows = Vec::new();
    let accepted = trainer
        .run_device_updates(&model, 2, 1, |indices| {
            retry_rows.extend_from_slice(indices);
            Ok(compute(true))
        })
        .unwrap();
    assert!(accepted.accepted);
    assert_eq!(trainer.progress().accepted, 2);
    assert_eq!(retry_rows, attempted_rows);
    println!("atomic objective rollback and replay cuda:{index}, attempts=2");
}

#[test]
fn candidate_overflow_rejects_the_whole_block_before_checkpoint_publication() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let device = Device::Cuda(0);
    let (vars, model, mut trainer) = fixture(device);
    let originals: BTreeMap<_, _> = vars
        .variables()
        .into_iter()
        .map(|(n, t)| (n, t.detach().copy()))
        .collect();
    let sample_before = trainer.sampler_snapshot();
    let mut i = 0;
    let err = trainer
        .run_device_updates(&model, 2, 1, |_| {
            let p = vars.variables().remove("stem.raw_matrix.weight").unwrap();
            let loss = if i == 0 {
                p.square().mean(Kind::Float)
            } else {
                (&p - p.detach()).sum(Kind::Float) * 1e30
            };
            i += 1;
            Ok(DeviceUpdateLoss {
                loss,
                valid: Tensor::ones([], (Kind::Bool, device)),
                group_activity: Tensor::ones([1], (Kind::Bool, device)),
            })
        })
        .unwrap_err();
    assert!(err.to_string().contains("candidate"), "{err}");
    assert_eq!(trainer.progress().attempted, 0);
    assert_eq!(trainer.sampler_snapshot(), sample_before);
    for (n, t) in vars.variables() {
        assert!(t.equal(&originals[&n]), "{n}");
    }
    assert!(trainer
        .step(1, |_| Ok(loss(&vars, 0, true)))
        .unwrap_err()
        .to_string()
        .contains("restore"));
}

fn logical_sample(vars: &VarStore, index: usize, position: usize) -> Tensor {
    let params = vars.variables();
    let base = &params["stem.raw_matrix.weight"];
    let group = &params["stem.group_roles.C02.0.weight"];
    let scale = 1.0 + (index + position) as f64 * 0.2;
    let mut result = (base * scale - 0.3).square().mean(Kind::Float);
    if (index + position) % 2 == 0 {
        result += (group * (scale + 0.4)).square().mean(Kind::Float);
    }
    result
}

#[test]
fn sequential_microbatches_match_one_logical_adamw_update() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let device = Device::Cuda(0);
    let (reference, ref_model, mut ref_trainer) = fixture(device);
    let (mut micro, model, mut trainer) = fixture(device);
    micro.copy(&reference).unwrap();
    let mut ref_rows = Vec::new();
    let ref_result = ref_trainer
        .run_device_updates(&ref_model, 2, 4, |indices| {
            ref_rows.extend_from_slice(indices);
            let losses: Vec<_> = indices
                .iter()
                .enumerate()
                .map(|(position, &i)| logical_sample(&reference, i, position))
                .collect();
            let active = indices.iter().enumerate().any(|(p, &i)| (i + p) % 2 == 0);
            Ok(DeviceUpdateLoss {
                loss: Tensor::stack(&losses, 0).mean(Kind::Float),
                valid: Tensor::ones([], (Kind::Bool, device)),
                group_activity: Tensor::full([1], i64::from(active), (Kind::Bool, device)),
            })
        })
        .unwrap();
    let mut rows = Vec::new();
    let result = trainer
        .run_device_microbatch_updates(&model, 2, 4, |indices, position| {
            assert_eq!(indices.len(), 1);
            rows.extend_from_slice(indices);
            let i = indices[0];
            Ok(DeviceUpdateLoss {
                loss: logical_sample(&micro, i, position),
                valid: Tensor::ones([], (Kind::Bool, device)),
                group_activity: Tensor::full(
                    [1],
                    i64::from((i + position) % 2 == 0),
                    (Kind::Bool, device),
                ),
            })
        })
        .unwrap();
    assert!(ref_result.accepted && result.accepted);
    assert_eq!(rows, ref_rows);
    assert_eq!(trainer.progress(), ref_trainer.progress());
    assert_eq!(trainer.sampler_snapshot(), ref_trainer.sampler_snapshot());
    let expected = reference.variables();
    for (name, p) in micro.variables() {
        assert!(
            p.allclose(&expected[&name], 3e-6, 3e-7, false),
            "parameter {name}"
        );
    }
    let (mh, moments) = saved(&micro, &trainer, "microbatch");
    let (rh, expected_moments) = saved(&reference, &ref_trainer, "whole-batch");
    assert_eq!(mh, rh);
    for (name, t) in moments {
        assert!(
            t.allclose(&expected_moments[&name], 3e-6, 2e-8, false),
            "moment {name}"
        );
    }
}

#[test]
fn sequential_microbatch_invalid_vote_rolls_back_and_replays_sampler() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let device = Device::Cuda(0);
    let (vars, model, mut trainer) = fixture(device);
    let initial: BTreeMap<_, _> = vars
        .variables()
        .into_iter()
        .map(|(name, t)| (name, t.detach().copy()))
        .collect();
    let sampler = trainer.sampler_snapshot();
    let mut attempted = Vec::new();
    let err = trainer
        .run_device_microbatch_updates(&model, 1, 4, |indices, position| {
            attempted.extend_from_slice(indices);
            Ok(DeviceUpdateLoss {
                loss: logical_sample(&vars, indices[0], position),
                valid: Tensor::full([], i64::from(position != 2), (Kind::Bool, device)),
                group_activity: Tensor::ones([1], (Kind::Bool, device)),
            })
        })
        .unwrap_err();
    assert!(err.to_string().contains("device objective"), "{err}");
    assert_eq!(trainer.progress().attempted, 0);
    assert_eq!(trainer.sampler_snapshot(), sampler);
    for (name, param) in vars.variables() {
        assert!(param.equal(&initial[&name]), "{name}");
    }
    let mut retry = Vec::new();
    trainer
        .run_device_microbatch_updates(&model, 1, 4, |indices, position| {
            retry.extend_from_slice(indices);
            Ok(DeviceUpdateLoss {
                loss: logical_sample(&vars, indices[0], position),
                valid: Tensor::ones([], (Kind::Bool, device)),
                group_activity: Tensor::ones([1], (Kind::Bool, device)),
            })
        })
        .unwrap();
    assert_eq!(attempted, retry);
    assert_eq!(trainer.progress().accepted, 1);
}
