//! Immutable, hash-checked training snapshots with atomic `last`/`best` pointers.
//!
//! Model, optimizer, and objective adapters own their binary encodings. The
//! store binds those bytes to the feature/data/objective contract and to the
//! exact sampler and progress state, then verifies all bytes before returning
//! a checkpoint to a caller.

use crate::training::{Progress, SamplerSnapshot, TrainingError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Training(TrainingError),
    Invalid(&'static str),
    AllocationFailed(&'static str),
    ContractMismatch,
    HashMismatch,
    ProgressRegression,
    ConflictingProgress,
}

impl Display for CheckpointError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "checkpoint I/O failed: {error}"),
            Self::Json(error) => write!(f, "checkpoint metadata is invalid: {error}"),
            Self::Training(error) => write!(f, "checkpoint training state is invalid: {error}"),
            Self::Invalid(reason) => write!(f, "invalid checkpoint: {reason}"),
            Self::AllocationFailed(which) => {
                write!(f, "could not reserve memory for checkpoint {which}")
            }
            Self::ContractMismatch => f.write_str("checkpoint contract differs from this run"),
            Self::HashMismatch => f.write_str("checkpoint hash does not match its contents"),
            Self::ProgressRegression => f.write_str("checkpoint progress cannot move backwards"),
            Self::ConflictingProgress => {
                f.write_str("different checkpoint contents already exist at this update")
            }
        }
    }
}

impl Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<TrainingError> for CheckpointError {
    fn from(error: TrainingError) -> Self {
        Self::Training(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub contract: Value,
    pub progress: Progress,
    pub sampler: SamplerSnapshot,
    pub model_state: Vec<u8>,
    pub optimizer_state: Vec<u8>,
    pub objective_state: Vec<u8>,
    pub rng_state: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    pub version: u32,
    pub file: String,
    pub sha256: String,
    pub contract_sha256: String,
    pub progress: Progress,
}

mod codec;
mod integrity;
mod store;

/// Checkpoint persistence rooted in one training run directory.
#[derive(Debug, Clone)]
pub struct CheckpointStore {
    root: PathBuf,
}
