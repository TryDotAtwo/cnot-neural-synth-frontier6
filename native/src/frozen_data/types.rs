//! Public data structures for sealed training artifacts.
use crate::state::MatrixState;
use crate::training_data::TrainingRecord;
use serde_json::Value;
use std::error::Error;
use std::fmt::{Display, Formatter};

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
    pub(super) records: Vec<TrainingRecord>,
    pub(super) metadata: Vec<FrozenRecordMetadata>,
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
    pub forbidden_keys: std::collections::HashSet<MatrixState>,
    pub historical_keys: std::collections::HashSet<MatrixState>,
    pub rejected_train_lineages: std::collections::HashSet<String>,
}
