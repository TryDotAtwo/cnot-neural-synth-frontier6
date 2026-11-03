//! Sealed reader for research_v3.data.frozen own_frozen_panels_v1 artifacts.
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use crate::training_data::{ExactCatalog, TrainingRecord, TrainingSplit};
#[cfg(test)]
use num_bigint::BigUint;
#[cfg(test)]
use serde_json::Number;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const SCHEMA: &str = "own_frozen_panels_v1";
const SYMMETRIES: &str =
    "identity,inverse,transpose,inverse_transpose and wire reversal; not full orbit";
const MAX_N: u32 = 256;

mod symmetry;
mod wire;
use symmetry::{disjoint, panel_views, validate_train, views};
use wire::{Body, key_set, panel};
#[cfg(test)]
use wire::{State, decode_state};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenDataError(pub String);
impl Display for FrozenDataError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for FrozenDataError {}

#[derive(Clone, Debug)]
pub struct FrozenRecordMetadata {
    pub source: String,
    pub seed: u64,
    pub exact_evidence: Option<String>,
}
#[derive(Clone, Debug)]
pub struct FrozenPanel {
    records: Vec<TrainingRecord>,
    metadata: Vec<FrozenRecordMetadata>,
}
impl FrozenPanel {
    pub fn records(&self) -> &[TrainingRecord] {
        &self.records
    }
    pub fn metadata(&self) -> &[FrozenRecordMetadata] {
        &self.metadata
    }
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
#[derive(Clone, Debug)]
pub struct FrozenDataset {
    pub train: FrozenPanel,
    pub dev: FrozenPanel,
    pub test: FrozenPanel,
    pub sha256: String,
    pub provenance: Value,
    pub forbidden_keys: HashSet<MatrixState>,
    pub historical_keys: HashSet<MatrixState>,
    pub rejected_train_lineages: HashSet<String>,
}

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

fn alphabet(s: &MatrixState) -> Result<CnotAlphabet, FrozenDataError> {
    CnotAlphabet::all_to_all_unit(s.dimension()).map_err(|e| FrozenDataError(e.to_string()))
}
fn split_name(s: TrainingSplit) -> &'static str {
    match s {
        TrainingSplit::Train => "train",
        TrainingSplit::Dev => "dev",
        TrainingSplit::Test => "test",
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
struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}
impl<R> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }
    fn finish(mut self) -> Result<String, FrozenDataError>
    where
        R: Read,
    {
        let mut buffer = [0u8; 1 << 16];
        loop {
            if self
                .read(&mut buffer)
                .map_err(|e| FrozenDataError(format!("finish hashing dataset: {e}")))?
                == 0
            {
                break;
            }
        }
        Ok(self
            .hasher
            .finalize()
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect())
    }
}
impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buf)?;
        self.hasher.update(&buf[..count]);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::str::FromStr;

    fn one_gate_record(n: u32, split: &str) -> (Value, MatrixState) {
        let alpha = CnotAlphabet::all_to_all_unit(n).unwrap();
        let identity = MatrixState::identity(n, alpha.id()).unwrap();
        let gate = alpha.gates()[0];
        let mut state = identity.clone();
        state.apply_cnot(gate).unwrap();
        let state_rows: Vec<_> = state
            .packed_rows()
            .chunks(state.words_per_row())
            .map(|row| {
                let bits = row
                    .iter()
                    .enumerate()
                    .fold(BigUint::from(0u8), |acc, (word, bits)| {
                        acc | (BigUint::from(*bits) << (word * 64))
                    });
                Number::from_str(&bits.to_str_radix(10)).unwrap()
            })
            .collect();
        let identity_rows: Vec<_> = identity
            .packed_rows()
            .chunks(identity.words_per_row())
            .map(|row| {
                let bits = row
                    .iter()
                    .enumerate()
                    .fold(BigUint::from(0u8), |acc, (word, bits)| {
                        acc | (BigUint::from(*bits) << (word * 64))
                    });
                Number::from_str(&bits.to_str_radix(10)).unwrap()
            })
            .collect();
        let lineage = format!("{split}-lineage");
        let record = json!({
            "state": {"n": n, "packed_rows": state_rows, "alphabet_id": alpha.id()},
            "verified_words": [{
                "start": {"n": n, "packed_rows": state.packed_rows().chunks(state.words_per_row()).map(|row| {
                    let bits = row.iter().enumerate().fold(BigUint::from(0u8), |acc, (word, bits)| acc | (BigUint::from(*bits) << (word * 64)));
                    Number::from_str(&bits.to_str_radix(10)).unwrap()
                }).collect::<Vec<_>>(), "alphabet_id": alpha.id()},
                "end": {"n": n, "packed_rows": identity_rows, "alphabet_id": alpha.id()},
                "gates": [{"control": gate.control, "target": gate.target}],
                "physical_cost": 1, "producer": "frozen-data-fixture", "lineage_id": lineage
            }],
            "lineage_id": lineage, "seed": n, "source": format!("source-{split}"),
            "split": split, "exact_distance": null, "exact_evidence": null
        });
        (record, state)
    }

    fn key_json(state: &MatrixState) -> Value {
        let rows: Vec<_> = state
            .packed_rows()
            .chunks(state.words_per_row())
            .map(|row| {
                let bits = row
                    .iter()
                    .enumerate()
                    .fold(BigUint::from(0u8), |acc, (word, bits)| {
                        acc | (BigUint::from(*bits) << (word * 64))
                    });
                Number::from_str(&bits.to_str_radix(10)).unwrap()
            })
            .collect();
        json!([state.dimension(), state.alphabet_id(), rows])
    }

    fn sealed_fixture(path: &Path) -> String {
        let (train, _) = one_gate_record(4, "train");
        let (dev, dev_state) = one_gate_record(3, "dev");
        let (test, test_state) = one_gate_record(2, "test");
        let mut forbidden = views(&dev_state).unwrap();
        forbidden.extend(views(&test_state).unwrap());
        let body = json!({
            "schema": SCHEMA,
            "provenance": {"fixture": "frozen_data_reader_test"},
            "symmetries": SYMMETRIES,
            "forbidden_keys": forbidden.iter().map(key_json).collect::<Vec<_>>(),
            "historical_keys": [],
            "rejected_train_lineages": [],
            "panels": {"train": [train], "dev": [dev], "test": [test]}
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        std::fs::write(path, &bytes).unwrap();
        let digest = bytes
            .iter()
            .fold(Sha256::new(), |mut h, byte| {
                h.update([*byte]);
                h
            })
            .finalize()
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect::<String>();
        std::fs::write(path.with_extension("sha256"), format!("{digest}\n")).unwrap();
        digest
    }

    #[test]
    fn pinned_load_verifies_split_artifact_and_rejects_wrong_pin() {
        let unique = format!(
            "tqp-frozen-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let digest = sealed_fixture(&path);
        let data = FrozenDataset::load_pinned(&path, &digest).unwrap();
        assert_eq!(data.sha256, digest);
        assert_eq!(data.train.len(), 1);
        assert_eq!(data.dev.len(), 1);
        assert_eq!(data.test.len(), 1);
        assert!(FrozenDataset::load_pinned(&path, &"0".repeat(64)).is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(path.with_extension("sha256")).unwrap();
    }

    #[cfg(feature = "torch")]
    #[test]
    fn objective_binding_uses_the_loaded_artifact_and_its_train_panel() {
        use crate::features::{FeaturePolicy, FeatureRegistry};
        use crate::objective::{FullGraphObjective, FullGraphObjectiveConfig};

        let unique = format!(
            "tqp-frozen-bind-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let pinned = sealed_fixture(&path);
        let dataset = FrozenDataset::load_pinned(&path, &pinned).unwrap();
        let registry = FeatureRegistry::new();
        let policy = FeaturePolicy::all_model_inputs("frozen-data-binding-fixture");
        let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
        let mut objective = FullGraphObjective::new(
            vec![alphabet],
            &registry,
            policy,
            Vec::new(),
            Vec::new(),
            FullGraphObjectiveConfig::default(),
        )
        .unwrap();

        objective.bind_frozen_dataset(&dataset).unwrap();
        let contract = objective.checkpoint_contract();
        assert_eq!(
            contract["frozen_dataset_sha256"].as_str(),
            Some(pinned.as_str())
        );
        assert!(contract["training_records_sha256"].as_str().is_some());
        assert!(contract["objective_sha256"].as_str().is_some());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(path.with_extension("sha256")).unwrap();
    }

    #[test]
    fn packed_rows_above_u64_keep_all_bits() {
        let n = 130u32;
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let rows: Vec<String> = (0..n)
            .map(|i| (BigUint::from(1u8) << i as usize).to_str_radix(10))
            .collect();
        let json = format!(
            "{{\"n\":{n},\"packed_rows\":[{}],\"alphabet_id\":\"{}\"}}",
            rows.join(","),
            a.id()
        );
        let raw: State = serde_json::from_str(&json).unwrap();
        let state = decode_state(raw).unwrap();
        assert_eq!(state.words_per_row(), 3);
        assert!(state.is_identity());
    }
    #[test]
    fn rejects_nonzero_padding_bits() {
        let raw: State = serde_json::from_str(
            r#"{"n":3,"packed_rows":[1,2,8],"alphabet_id":"all_to_all_unit_cnot:3"}"#,
        )
        .unwrap();
        assert!(
            decode_state(raw)
                .unwrap_err()
                .0
                .contains("outside matrix width")
        );
    }
}
