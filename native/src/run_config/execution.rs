use super::manifest::{parse_run_manifest, AlphabetSpec};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::frozen_data::FrozenDataset;
use crate::model::FusionDistanceField;
use crate::nn_training::NeuralTrainer;
use crate::objective::{ExposureAudit, FullGraphObjective};
use crate::training::Sampler;
use crate::training_session::{
    FullGraphTrainingSession, TrainingSessionConfig, TrainingSessionResources,
};
use crate::{load_exact_catalogs_pinned, MatrixState};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use tch::{nn::VarStore, Device};

fn resolve_path(manifest: &Path, configured: &Path) -> PathBuf {
    if configured.is_absolute() {
        configured.to_owned()
    } else {
        manifest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(configured)
    }
}
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn execute_run_manifest(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path)
        .map_err(|e| format!("could not read run manifest {}: {e}", path.display()))?;
    let manifest = parse_run_manifest(&raw)?;
    let run_sha256 = sha256(&raw);
    let dataset_path = resolve_path(path, &manifest.dataset.path);
    let catalog_path = resolve_path(path, &manifest.exact_catalogs.path);
    let checkpoint_dir = resolve_path(path, &manifest.checkpoint_dir);
    if manifest.resume {
        if !checkpoint_dir.join("last.json").is_file() {
            return Err(format!(
                "resume requested but {} has no last.json",
                checkpoint_dir.display()
            ));
        }
    } else if checkpoint_dir.exists()
        && fs::read_dir(&checkpoint_dir)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "fresh run refuses nonempty checkpoint directory {}",
            checkpoint_dir.display()
        ));
    }

    crate::torch_device::require_cuda_device(manifest.training.cuda_device)?;
    let alphabets = manifest
        .alphabets
        .iter()
        .map(AlphabetSpec::build)
        .collect::<Result<Vec<_>, _>>()?;
    let mut ids = HashSet::with_capacity(alphabets.len());
    for alphabet in &alphabets {
        if !ids.insert(alphabet.id().to_owned()) {
            return Err(format!("duplicate alphabet identity {}", alphabet.id()));
        }
        let canonical =
            CnotAlphabet::all_to_all_unit(alphabet.dimension()).map_err(|e| e.to_string())?;
        if alphabet.id() != canonical.id()
            || alphabet.gates() != canonical.gates()
            || (0..alphabet.gates().len()).any(|i| alphabet.cost_at(i) != canonical.cost_at(i))
        {
            return Err(format!("pinned exact-catalog bundle supports only all-to-all unit alphabet {}; supplied {}", canonical.id(), alphabet.id()));
        }
    }
    let catalogs = load_exact_catalogs_pinned(&catalog_path, &manifest.exact_catalogs.sha256)
        .map_err(|e| e.to_string())?;
    if catalogs.len() != alphabets.len() || alphabets.iter().any(|a| !catalogs.contains_key(a.id()))
    {
        return Err("exact catalogs must cover exactly the configured alphabets".into());
    }
    let dataset = FrozenDataset::load_pinned_with_catalogs(
        &dataset_path,
        &manifest.dataset.sha256,
        &catalogs,
    )
    .map_err(|e| e.to_string())?;
    let feature_contexts = manifest.feature_contexts.build()?;
    let profile_alphabet = &feature_contexts.alphabet;
    if !alphabets.iter().any(|a| {
        a.id() == profile_alphabet.id()
            && a.dimension() == profile_alphabet.dimension()
            && a.gates() == profile_alphabet.gates()
            && (0..a.gates().len()).all(|i| a.cost_at(i) == profile_alphabet.cost_at(i))
    }) {
        return Err("feature-context alphabet must exactly match an objective alphabet".into());
    }
    let registry = FeatureRegistry::with_all_model_input_providers(feature_contexts)
        .map_err(|e| e.to_string())?;
    let mut policy = FeaturePolicy::all_model_inputs_augmentation_only(
        manifest.feature_policy.schema_id.clone(),
    );
    policy.seed = manifest.training.seed;
    policy.solver_assisted = manifest.feature_policy.solver_assisted;
    policy.evaluations = manifest.feature_policy.evaluations.into_option();
    policy.max_bytes = manifest.feature_policy.max_bytes.into_option();
    let selected = registry
        .selected_specs(&policy)
        .map_err(|e| e.to_string())?;
    if selected.len() != crate::schema::MODEL_INPUT_COUNT {
        return Err(format!(
            "expected all {} model inputs, registry selected {}",
            crate::schema::MODEL_INPUT_COUNT,
            selected.len()
        ));
    }
    let exact_catalogs = catalogs.values().cloned().collect();
    let forbidden_states = dataset
        .forbidden_keys
        .iter()
        .cloned()
        .collect::<Vec<MatrixState>>();
    let mut objective = FullGraphObjective::new(
        alphabets,
        &registry,
        policy.clone(),
        exact_catalogs,
        forbidden_states,
        manifest.objective.build(),
    )
    .map_err(|e| format!("objective construction failed: {e}"))?;
    tch::manual_seed(manifest.training.seed as i64);
    let device = Device::Cuda(manifest.training.cuda_device);
    let mut variables = VarStore::new(device);
    let model = FusionDistanceField::new_for_policy(
        &variables.root(),
        &registry,
        &policy,
        device,
        manifest.model.build(),
    )
    .map_err(|e| e.to_string())?;
    let sampler =
        Sampler::new(dataset.train.len(), manifest.training.seed).map_err(|e| e.to_string())?;
    let mut trainer =
        NeuralTrainer::new(&variables, sampler, manifest.optimizer).map_err(|e| e.to_string())?;
    let store = crate::CheckpointStore::new(&checkpoint_dir);
    let session_config = TrainingSessionConfig {
        batch_size: manifest.training.batch_size,
        checkpoint_every_attempts: manifest.training.checkpoint_every_attempts,
    };
    let resources = TrainingSessionResources {
        dataset: &dataset,
        objective: &mut objective,
        model: &model,
        variables: &mut variables,
        registry: &registry,
        trainer: &mut trainer,
        store: &store,
        admission: None,
    };
    let report = {
        let mut session = if manifest.resume {
            FullGraphTrainingSession::resume_latest(resources, session_config)
                .map_err(|e| format!("checkpoint resume failed: {e}"))?
        } else {
            FullGraphTrainingSession::new(resources, session_config)
                .map_err(|e| format!("new training session failed: {e}"))?
        };
        session
            .run_attempts(manifest.training.attempts)
            .map_err(|e| format!("training session failed: {e}"))?
    };
    let audit = objective.audit();
    if !audit.strict_unseen {
        return Err("post-run exposure audit found overlap with forbidden states".into());
    }
    let loss = if report.last_step.loss.is_finite() {
        Value::from(report.last_step.loss)
    } else {
        Value::Null
    };
    Ok(json!({
        "schema": "cnot103_training_result_v1",
        "run_manifest_sha256": run_sha256,
        "dataset_sha256": dataset.sha256,
        "exact_catalog_bundle_sha256": manifest.exact_catalogs.sha256,
        "objective_sha256": objective.config_sha256(),
        "feature_registry_sha256": registry.contract_sha256(),
        "feature_model_inputs": selected.len(),
        "train_rows": dataset.train.len(),
        "dev_rows": dataset.dev.len(),
        "test_rows": dataset.test.len(),
        "start_progress": report.start_progress,
        "end_progress": report.end_progress,
        "last_step": { "accepted": report.last_step.accepted, "loss": loss },
        "exposure_audit": audit_json(audit),
        "checkpoint": report.final_checkpoint,
    }))
}

fn audit_json(a: ExposureAudit) -> Value {
    json!({
        "strict_unseen": a.strict_unseen, "audit_mode": a.audit_mode,
        "label_count": a.label_count, "forward_count": a.forward_count, "gradient_count": a.gradient_count,
        "forbidden_count": a.forbidden_count, "label_overlap": a.label_overlap,
        "forward_overlap": a.forward_overlap, "gradient_overlap": a.gradient_overlap,
        "set_sha256": a.set_sha256
    })
}
