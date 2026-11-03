//! Epoch permutation sampling with exact continuation snapshots.
use super::python_random::{PythonRandom, PythonRandomSnapshot};
use super::TrainingError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerSnapshot {
    pub size: usize,
    pub order: Vec<usize>,
    pub cursor: usize,
    pub epochs: u64,
    pub rng: PythonRandomSnapshot,
}

/// Epoch permutation sampler matching the reference Python training schedule.
#[derive(Debug, Clone)]
pub struct Sampler {
    size: usize,
    order: Vec<usize>,
    cursor: usize,
    epochs: u64,
    rng: PythonRandom,
}

impl Sampler {
    pub fn new(size: usize, seed: u64) -> Result<Self, TrainingError> {
        if size == 0 {
            return Err(TrainingError::InvalidDatasetSize);
        }
        Ok(Self {
            size,
            order: Vec::new(),
            cursor: 0,
            epochs: 0,
            rng: PythonRandom::from_seed(seed),
        })
    }

    /// Return exactly `count` indices, crossing epoch boundaries if needed.
    pub fn next(&mut self, count: usize) -> Result<Vec<usize>, TrainingError> {
        if count == 0 {
            return Err(TrainingError::InvalidBatchSize);
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| TrainingError::AllocationFailed)?;
        while result.len() < count {
            if self.cursor == self.order.len() {
                let epochs = self
                    .epochs
                    .checked_add(1)
                    .ok_or(TrainingError::CounterOverflow)?;
                let mut order: Vec<usize> = (0..self.size).collect();
                self.rng.shuffle(&mut order);
                self.order = order;
                self.cursor = 0;
                self.epochs = epochs;
            }
            let remaining = count - result.len();
            let take = remaining.min(self.size - self.cursor);
            result.extend_from_slice(&self.order[self.cursor..self.cursor + take]);
            self.cursor += take;
        }
        Ok(result)
    }

    pub fn snapshot(&self) -> SamplerSnapshot {
        SamplerSnapshot {
            size: self.size,
            order: self.order.clone(),
            cursor: self.cursor,
            epochs: self.epochs,
            rng: self.rng.snapshot(),
        }
    }

    /// Validate the full snapshot before mutating the sampler.
    pub fn restore(&mut self, snapshot: &SamplerSnapshot) -> Result<(), TrainingError> {
        if snapshot.size != self.size {
            return Err(TrainingError::InvalidSamplerState("dataset size mismatch"));
        }
        if snapshot.cursor > snapshot.order.len() {
            return Err(TrainingError::InvalidSamplerState(
                "cursor exceeds permutation",
            ));
        }
        if snapshot.order.is_empty() {
            if snapshot.cursor != 0 || snapshot.epochs != 0 {
                return Err(TrainingError::InvalidSamplerState(
                    "empty permutation has progress",
                ));
            }
        } else if snapshot.order.len() != self.size || !is_permutation(&snapshot.order, self.size) {
            return Err(TrainingError::InvalidSamplerState(
                "order is not a dataset permutation",
            ));
        } else if snapshot.epochs == 0 {
            return Err(TrainingError::InvalidSamplerState(
                "permutation exists before the first epoch",
            ));
        }
        let restored_rng = PythonRandom::from_snapshot(&snapshot.rng)?;

        self.order.clone_from(&snapshot.order);
        self.cursor = snapshot.cursor;
        self.epochs = snapshot.epochs;
        self.rng = restored_rng;
        Ok(())
    }
}

fn is_permutation(order: &[usize], size: usize) -> bool {
    if order.len() != size {
        return false;
    }
    let mut seen = vec![false; size];
    for &index in order {
        if index >= size || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}
