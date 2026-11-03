use serde_json::Value;

use super::{RawRunManifest, RunManifest, RUN_SCHEMA};

pub fn parse_run_manifest(raw: &[u8]) -> Result<RunManifest, String> {
    let value: Value = serde_json::from_slice(raw).map_err(|e| format!("invalid run JSON: {e}"))?;
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| "run manifest requires a string schema".to_owned())?;
    if schema != RUN_SCHEMA {
        return Err(format!(
            "run manifest schema must be {RUN_SCHEMA}, got {schema}"
        ));
    }
    let raw: RawRunManifest =
        serde_json::from_value(value).map_err(|e| format!("invalid run manifest: {e}"))?;
    if raw.schema != RUN_SCHEMA {
        return Err(format!("run manifest schema must be {RUN_SCHEMA}"));
    }
    let manifest = RunManifest {
        dataset: raw.dataset,
        exact_catalogs: raw.exact_catalogs,
        checkpoint_dir: raw.checkpoint_dir,
        resume: raw.resume,
        training: raw.training,
        feature_policy: raw.feature_policy,
        model: raw.model,
        optimizer: raw.optimizer,
        objective: raw.objective,
        alphabets: raw.alphabets,
        feature_contexts: raw.feature_contexts,
    };
    manifest.validate()?;
    Ok(manifest)
}
