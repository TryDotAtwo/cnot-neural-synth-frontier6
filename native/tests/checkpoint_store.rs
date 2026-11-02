use cnot103_native::{Checkpoint, CheckpointStore, Progress, Sampler, SamplerSnapshot};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
const CHILD_DIR_ENV: &str = "CNOT103_CHECKPOINT_CHILD_DIR";

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "cnot103-checkpoint-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn contract() -> Value {
    json!({
        "model": {"id": "fixture-v1", "features": 103},
        "dataset_sha256": "d".repeat(64),
        "objective": {"id": "test-objective", "weight": 0.5}
    })
}

fn checkpoint(sampler: SamplerSnapshot) -> Checkpoint {
    Checkpoint {
        contract: contract(),
        progress: Progress {
            attempted: 2,
            accepted: 1,
            rejected: 1,
        },
        sampler,
        model_state: b"model-state".to_vec(),
        optimizer_state: b"optimizer-state".to_vec(),
        objective_state: b"objective-state".to_vec(),
        rng_state: b"runtime-rng-state".to_vec(),
    }
}

#[test]
fn checkpoint_is_immutable_and_round_trips_exact_state() {
    let directory = TempDir::new();
    let store = CheckpointStore::new(&directory.0);
    let mut sampler = Sampler::new(7, 42).unwrap();
    sampler.next(5).unwrap();
    sampler.next(2).unwrap();
    let snapshot = sampler.snapshot();
    let original = sampler.next(9).unwrap();
    let state = checkpoint(snapshot.clone());

    let manifest = store.save(&state, true).unwrap();
    let repeated = store.save(&state, false).unwrap();
    assert_eq!(manifest, repeated);
    assert!(directory.0.join("best.json").is_file());
    assert_eq!(store.load_latest(&contract()).unwrap(), state);
    assert_eq!(store.load_best(&contract()).unwrap(), state);

    let mut resumed = Sampler::new(7, 999).unwrap();
    resumed.restore(&snapshot).unwrap();
    assert_eq!(resumed.next(9).unwrap(), original);
}

#[test]
fn idempotent_save_validates_and_repairs_snapshot_and_honors_best_pointer() {
    let directory = TempDir::new();
    let store = CheckpointStore::new(&directory.0);
    let mut sampler = Sampler::new(5, 13).unwrap();
    sampler.next(3).unwrap();
    let state = checkpoint(sampler.snapshot());

    let manifest = store.save(&state, false).unwrap();
    assert!(!directory.0.join("best.json").exists());

    // Repeating the same update with update_best=true must still publish best.
    assert_eq!(store.save(&state, true).unwrap(), manifest);
    assert!(directory.0.join("best.json").is_file());

    // If the immutable payload disappeared, an idempotent save must restore it
    // instead of returning a manifest that points to a missing checkpoint.
    fs::remove_file(directory.0.join(&manifest.file)).unwrap();
    assert_eq!(store.save(&state, true).unwrap(), manifest);
    assert_eq!(store.load_latest(&contract()).unwrap(), state);
    assert_eq!(store.load_best(&contract()).unwrap(), state);
}

#[test]
fn checkpoint_refuses_contract_mismatch_conflict_and_corruption() {
    let directory = TempDir::new();
    let store = CheckpointStore::new(&directory.0);
    let mut sampler = Sampler::new(3, 7).unwrap();
    sampler.next(2).unwrap();
    let state = checkpoint(sampler.snapshot());
    let manifest = store.save(&state, false).unwrap();

    assert!(store
        .load_manifest(&manifest, &json!({"different": true}))
        .is_err());

    let mut conflict = state.clone();
    conflict.model_state.push(0);
    assert!(store.save(&conflict, false).is_err());

    let path = directory.0.join(&manifest.file);
    use std::io::Write;
    fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap()
        .write_all(b"tamper")
        .unwrap();
    assert!(store.load_latest(&contract()).is_err());
    assert!(store.save(&state, false).is_err());
}

#[test]
fn checkpoint_refuses_progress_regression() {
    let directory = TempDir::new();
    let store = CheckpointStore::new(&directory.0);
    let mut sampler = Sampler::new(3, 8).unwrap();
    sampler.next(2).unwrap();
    let mut state = checkpoint(sampler.snapshot());
    store.save(&state, false).unwrap();

    state.progress = Progress {
        attempted: 1,
        accepted: 1,
        rejected: 0,
    };
    assert!(store.save(&state, false).is_err());
}

#[test]
fn checkpoint_publishes_new_last_without_mutating_the_previous_snapshot() {
    let directory = TempDir::new();
    let store = CheckpointStore::new(&directory.0);
    let mut sampler = Sampler::new(3, 8).unwrap();
    sampler.next(2).unwrap();
    let mut state = checkpoint(sampler.snapshot());
    let first = store.save(&state, true).unwrap();

    state.progress = Progress {
        attempted: 3,
        accepted: 2,
        rejected: 1,
    };
    state.model_state.push(7);
    let second = store.save(&state, false).unwrap();

    assert_eq!(store.load_latest(&contract()).unwrap(), state);
    assert_eq!(store.load_best(&contract()).unwrap().progress.attempted, 2);
    assert!(directory.0.join(first.file).is_file());
    assert!(directory.0.join(second.file).is_file());
}

#[test]
fn checkpoint_fresh_process_child() {
    let Ok(directory) = std::env::var(CHILD_DIR_ENV) else {
        return;
    };
    let store = CheckpointStore::new(directory);
    let checkpoint = store.load_latest(&contract()).unwrap();
    let mut sampler = Sampler::new(checkpoint.sampler.size, 0).unwrap();
    sampler.restore(&checkpoint.sampler).unwrap();
    let next = sampler.next(5).unwrap();
    fs::write(
        PathBuf::from(std::env::var(CHILD_DIR_ENV).unwrap()).join("child-batch.json"),
        serde_json::to_vec(&next).unwrap(),
    )
    .unwrap();
}

#[test]
fn checkpoint_restores_in_a_fresh_process() {
    let directory = TempDir::new();
    let store = CheckpointStore::new(&directory.0);
    let mut sampler = Sampler::new(11, 2026).unwrap();
    sampler.next(7).unwrap();
    let snapshot = sampler.snapshot();
    let expected = sampler.next(5).unwrap();
    let state = checkpoint(snapshot);
    store.save(&state, false).unwrap();

    let output = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("checkpoint_fresh_process_child")
        .arg("--nocapture")
        .env(CHILD_DIR_ENV, &directory.0)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let restored: Vec<usize> =
        serde_json::from_slice(&fs::read(directory.0.join("child-batch.json")).unwrap()).unwrap();
    assert_eq!(restored, expected);
}
