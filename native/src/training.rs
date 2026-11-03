//! Deterministic training schedule primitives.
//!
//! The sampler preserves CPython seed-to-order behavior so prior Python runs can
//! resume without changing their training data sequence.
use std::error::Error;
use std::fmt::{Display, Formatter};

mod progress;
mod python_random;
mod sampler;

pub use progress::Progress;
pub use python_random::PythonRandomSnapshot;
pub use sampler::{Sampler, SamplerSnapshot};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingError {
    InvalidDatasetSize,
    InvalidBatchSize,
    AllocationFailed,
    CounterOverflow,
    InvalidProgress,
    InvalidSamplerState(&'static str),
}

impl Display for TrainingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDatasetSize => f.write_str("dataset size must be positive"),
            Self::InvalidBatchSize => f.write_str("batch size must be positive"),
            Self::AllocationFailed => f.write_str("could not reserve sampler batch storage"),
            Self::CounterOverflow => f.write_str("training progress counter overflow"),
            Self::InvalidProgress => f.write_str("training progress counters are inconsistent"),
            Self::InvalidSamplerState(reason) => write!(f, "invalid sampler state: {reason}"),
        }
    }
}

impl Error for TrainingError {}
