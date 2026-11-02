//! Regression coverage for fresh-process restoration and optimizer atomicity.

use cnot103_native::nn_training::{AdamWConfig, NeuralTrainer, TchTrainingError};
use cnot103_native::{CheckpointStore, Progress, Sampler};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tch::nn::{self, VarStore};
use tch::{Device, Kind, Tensor};

const DATASET_SIZE: usize = 11;
const BATCH_SIZE: usize = 3;
const SPLIT_COUNT: usize = 4;
const TOTAL_COUNT: usize = 8;
const SEED: u64 = 2026;
const CHILD_STORE_ENV: &str = "CNOT103_RESUME_CHILD_STORE";
const CHILD_OUTPUT_ENV: &str = "CNOT103_RESUME_CHILD_OUTPUT";

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before UNIX epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{label}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn contract() -> Value {
    json!({
        "test": "tch_training_resume",
        "version": 1
    })
}

fn test_device() -> Device {
    match std::env::var("CNOT103_RESUME_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok("cuda:0") => {
            cnot103_native::torch_device::require_cuda_device(0)
                .expect("initialize CUDA for checkpoint-resume test");
            Device::Cuda(0)
        }
        Ok("cuda:1") => {
            cnot103_native::torch_device::require_cuda_device(1)
                .expect("initialize second CUDA device for checkpoint-resume test");
            Device::Cuda(1)
        }
        Ok(other) => panic!("unsupported CNOT103_RESUME_DEVICE value: {other}"),
    }
}

fn new_run() -> (VarStore, Tensor, NeuralTrainer) {
    let mut variables = VarStore::new(test_device());
    variables.set_kind(Kind::Double);
    let weight = variables.root().var("weight", &[1], nn::Init::Const(0.25));
    let mut trainer = NeuralTrainer::new(
        &variables,
        Sampler::new(DATASET_SIZE, SEED).expect("create sampler"),
        AdamWConfig::default(),
    )
    .expect("create trainer");
    trainer.set_objective_state(b"objective-state-v1".to_vec());
    trainer.set_rng_state(b"caller-rng-state-v1".to_vec());
    (variables, weight, trainer)
}

fn run_updates(trainer: &mut NeuralTrainer, weight: &Tensor, count: usize) {
    for _ in 0..count {
        let outcome = trainer
            .step(BATCH_SIZE, |indices| {
                let target =
                    indices.iter().map(|index| *index as f64).sum::<f64>() / indices.len() as f64;
                Ok((weight - target).pow_tensor_scalar(2.0).mean(Kind::Double))
            })
            .expect("training update");
        assert!(outcome.accepted);
        assert!(outcome.loss.is_finite());
    }
}

fn summary(trainer: &NeuralTrainer, weight: &Tensor) -> Value {
    json!({
        "weight": weight.double_value(&[0]),
        "progress": trainer.progress(),
        "optimizer_steps": trainer.optimizer_steps(),
        "sampler": trainer.sampler_snapshot(),
        "objective_state": trainer.objective_state(),
        "rng_state": trainer.rng_state(),
    })
}

#[test]
fn fresh_process_resume_matches_uninterrupted() {
    let temp = TestDir::new("cnot103-fresh-resume");
    let store = CheckpointStore::new(temp.path().join("checkpoint"));
    let expected_contract = contract();

    let (variables, weight, mut uninterrupted) = new_run();
    run_updates(&mut uninterrupted, &weight, TOTAL_COUNT);
    let expected = summary(&uninterrupted, &weight);

    let (variables_before_split, weight_before_split, mut split) = new_run();
    run_updates(&mut split, &weight_before_split, SPLIT_COUNT);
    split
        .save_checkpoint(&variables_before_split, &store, expected_contract, false)
        .expect("save split checkpoint");

    let output_path = temp.path().join("fresh-process-result.json");
    let child = Command::new(std::env::current_exe().expect("test executable path"))
        .arg("--ignored")
        .arg("--exact")
        .arg("fresh_process_resume_child")
        .arg("--nocapture")
        .env(CHILD_STORE_ENV, store.root())
        .env(CHILD_OUTPUT_ENV, &output_path)
        .output()
        .expect("launch fresh process");
    assert!(
        child.status.success(),
        "fresh process failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );

    let resumed: Value =
        serde_json::from_slice(&fs::read(output_path).expect("read fresh-process summary"))
            .expect("parse fresh-process summary");
    assert_eq!(resumed, expected);
    drop(variables);
}

#[test]
#[ignore = "invoked in a separate process by fresh_process_resume_matches_uninterrupted"]
fn fresh_process_resume_child() {
    let (Some(store_path), Some(output_path)) = (
        std::env::var_os(CHILD_STORE_ENV),
        std::env::var_os(CHILD_OUTPUT_ENV),
    ) else {
        return;
    };

    let (mut variables, weight, mut trainer) = new_run();
    let store = CheckpointStore::new(store_path);
    trainer
        .restore_checkpoint(&mut variables, &store, &contract())
        .expect("restore checkpoint in fresh process");
    run_updates(&mut trainer, &weight, TOTAL_COUNT - SPLIT_COUNT);

    fs::write(
        output_path,
        serde_json::to_vec(&summary(&trainer, &weight)).expect("encode summary"),
    )
    .expect("write fresh-process summary");
}

#[test]
fn finite_loss_with_overflowing_adam_moments_requires_restore() {
    let temp = TestDir::new("cnot103-overflow");
    let store = CheckpointStore::new(temp.path().join("checkpoint"));
    let expected_contract = contract();

    let (mut variables, weight, mut trainer) = new_run();
    trainer
        .save_checkpoint(&variables, &store, expected_contract.clone(), false)
        .expect("save pre-failure checkpoint");
    let before = weight.double_value(&[0]);

    let result = trainer.step(1, |_| Ok((&weight * 1e200_f64).sum(Kind::Double)));
    assert!(
        matches!(result, Err(TchTrainingError::Invalid(_))),
        "finite loss with an overflowing Adam moment must reject the optimizer update"
    );
    assert_eq!(trainer.progress(), Progress::default());
    assert_eq!(trainer.optimizer_steps(), 0);
    assert_eq!(weight.double_value(&[0]), before);
    assert!(trainer
        .save_checkpoint(&variables, &store, expected_contract.clone(), false)
        .is_err());

    let blocked = trainer.step(1, |_| {
        panic!("poisoned trainer must refuse work before evaluating the objective")
    });
    assert!(matches!(blocked, Err(TchTrainingError::Invalid(_))));

    trainer
        .restore_checkpoint(&mut variables, &store, &expected_contract)
        .expect("restore clean pre-failure checkpoint");
    assert_eq!(trainer.progress(), Progress::default());
    assert_eq!(trainer.optimizer_steps(), 0);
    let outcome = trainer
        .step(1, |_| {
            Ok((&weight - 0.5_f64).pow_tensor_scalar(2.0).sum(Kind::Double))
        })
        .expect("valid step after restore");
    assert!(outcome.accepted);
    assert_eq!(trainer.progress().accepted, 1);
    assert_eq!(trainer.optimizer_steps(), 1);
}
