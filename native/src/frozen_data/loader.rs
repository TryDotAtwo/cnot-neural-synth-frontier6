//! Pinned loading and semantic validation for sealed frozen-data artifacts.
use super::reader::HashingReader;
use super::symmetry::{disjoint, panel_views, validate_train, views};
use super::types::{FrozenDataError, FrozenDataset};
use super::wire::{key_set, panel, Body};
use crate::training_data::{ExactCatalog, TrainingSplit};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub(super) const SCHEMA: &str = "own_frozen_panels_v1";
pub(super) const SYMMETRIES: &str =
    "identity,inverse,transpose,inverse_transpose and wire reversal; not full orbit";

impl FrozenDataset {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FrozenDataError> {
        Self::load_with_catalogs(path, &HashMap::new())
    }

    pub fn load_with_catalogs(
        path: impl AsRef<Path>,
        catalogs: &HashMap<String, ExactCatalog>,
    ) -> Result<Self, FrozenDataError> {
        Self::load_internal(path.as_ref(), None, catalogs)
    }

    /// Load a sealed artifact and also require its digest to match a trusted pin
    /// supplied by the caller (for example, a versioned experiment manifest).
    pub fn load_pinned(
        path: impl AsRef<Path>,
        pinned_sha256: &str,
    ) -> Result<Self, FrozenDataError> {
        Self::load_pinned_with_catalogs(path, pinned_sha256, &HashMap::new())
    }

    pub fn load_pinned_with_catalogs(
        path: impl AsRef<Path>,
        pinned_sha256: &str,
        catalogs: &HashMap<String, ExactCatalog>,
    ) -> Result<Self, FrozenDataError> {
        if !is_sha256(pinned_sha256) {
            return Err(FrozenDataError(
                "pinned frozen-data digest must be lowercase SHA-256 hex".into(),
            ));
        }
        Self::load_internal(path.as_ref(), Some(pinned_sha256), catalogs)
    }

    fn load_internal(
        path: &Path,
        pinned_sha256: Option<&str>,
        catalogs: &HashMap<String, ExactCatalog>,
    ) -> Result<Self, FrozenDataError> {
        let seal_path = path.with_extension("sha256");
        let seal = std::fs::read_to_string(&seal_path)
            .map_err(|e| FrozenDataError(format!("could not read {}: {e}", seal_path.display())))?;
        let sealed = seal.trim();
        if !is_sha256(sealed) {
            return Err(FrozenDataError(
                "frozen-data sidecar must contain lowercase SHA-256 hex".into(),
            ));
        }
        let file = File::open(path).map_err(|e| FrozenDataError(format!("open dataset: {e}")))?;
        let mut reader = HashingReader::new(BufReader::new(file));
        let body: Body = serde_json::from_reader(&mut reader)
            .map_err(|e| FrozenDataError(format!("invalid frozen-data JSON: {e}")))?;
        let actual = reader.finish()?;
        if sealed != actual {
            return Err(FrozenDataError(format!(
                "frozen-data SHA-256 mismatch: expected {sealed}, got {actual}"
            )));
        }
        if let Some(pinned) = pinned_sha256 {
            if pinned != actual {
                return Err(FrozenDataError(format!(
                    "frozen-data digest differs from trusted pin: expected {pinned}, got {actual}"
                )));
            }
        }
        Self::from_body(body, actual, catalogs)
    }

    fn from_body(
        body: Body,
        sha256: String,
        catalogs: &HashMap<String, ExactCatalog>,
    ) -> Result<Self, FrozenDataError> {
        if body.schema != SCHEMA {
            return Err(FrozenDataError("schema mismatch".into()));
        }
        if !truthy(&body.provenance) {
            return Err(FrozenDataError("empty provenance".into()));
        }
        if body.symmetries != SYMMETRIES {
            return Err(FrozenDataError("unknown finite-symmetry contract".into()));
        }
        let rejected: HashSet<_> = body.rejected_train_lineages.iter().cloned().collect();
        if rejected.len() != body.rejected_train_lineages.len()
            || rejected.iter().any(|x| x.trim().is_empty())
        {
            return Err(FrozenDataError(
                "invalid rejected train-lineage list".into(),
            ));
        }
        let mut units = HashMap::new();
        let train = panel(
            body.panels.train,
            TrainingSplit::Train,
            "train",
            catalogs,
            &mut units,
            &rejected,
        )?;
        let dev = panel(
            body.panels.dev,
            TrainingSplit::Dev,
            "dev",
            catalogs,
            &mut units,
            &rejected,
        )?;
        let test = panel(
            body.panels.test,
            TrainingSplit::Test,
            "test",
            catalogs,
            &mut units,
            &rejected,
        )?;
        if train.is_empty() || dev.is_empty() || test.is_empty() {
            return Err(FrozenDataError(
                "train/dev/test panels must be nonempty".into(),
            ));
        }

        let historical = key_set(body.historical_keys, "historical_keys")?;
        let forbidden = key_set(body.forbidden_keys, "forbidden_keys")?;
        let dev_views = panel_views(&dev)?;
        let test_views = panel_views(&test)?;
        disjoint(&dev_views, &test_views, "dev/test finite-symmetry overlap")?;
        disjoint(&historical, &dev_views, "historical/dev overlap")?;
        disjoint(&historical, &test_views, "historical/test overlap")?;
        for state in &historical {
            if !views(state)?.is_subset(&historical) {
                return Err(FrozenDataError(
                    "historical keys are not symmetry-closed".into(),
                ));
            }
        }
        let mut expected = historical.clone();
        expected.extend(dev_views);
        expected.extend(test_views);
        if expected != forbidden {
            return Err(FrozenDataError(
                "forbidden-key set differs from frozen split roots".into(),
            ));
        }
        validate_train(&train, &forbidden)?;
        Ok(Self {
            train,
            dev,
            test,
            sha256,
            provenance: body.provenance,
            forbidden_keys: forbidden,
            historical_keys: historical,
            rejected_train_lineages: rejected,
        })
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Array(x) => !x.is_empty(),
        Value::Object(x) => !x.is_empty(),
        Value::String(x) => !x.trim().is_empty(),
        _ => true,
    }
}
fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
