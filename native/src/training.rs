//! Deterministic training schedule primitives.
//!
//! `Sampler` deliberately reproduces CPython's `random.Random(seed).shuffle`
//! sequence for non-negative `u64` seeds. This keeps existing Python training
//! runs resumable without silently changing their sample order.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

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

/// Attempt, acceptance, and rejection counts for guarded optimizer steps.
///
/// The invariant `attempted == accepted + rejected` is maintained by
/// [`record_attempt`](Self::record_attempt), using checked arithmetic.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub attempted: u64,
    pub accepted: u64,
    pub rejected: u64,
}

impl Progress {
    pub fn record_attempt(&mut self, accepted: bool) -> Result<(), TrainingError> {
        if !self.validate() {
            return Err(TrainingError::InvalidProgress);
        }
        let attempted = self
            .attempted
            .checked_add(1)
            .ok_or(TrainingError::CounterOverflow)?;
        let (new_accepted, new_rejected) = if accepted {
            (
                self.accepted
                    .checked_add(1)
                    .ok_or(TrainingError::CounterOverflow)?,
                self.rejected,
            )
        } else {
            (
                self.accepted,
                self.rejected
                    .checked_add(1)
                    .ok_or(TrainingError::CounterOverflow)?,
            )
        };
        self.attempted = attempted;
        self.accepted = new_accepted;
        self.rejected = new_rejected;
        Ok(())
    }

    pub fn validate(&self) -> bool {
        self.accepted.checked_add(self.rejected) == Some(self.attempted)
    }
}

/// Serializable MT19937 state compatible with CPython's `random.Random`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonRandomSnapshot {
    pub words: Vec<u32>,
    /// Number of already-consumed words. CPython uses 624 for a freshly seeded
    /// generator and 0..623 after draws.
    pub index: usize,
}

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

#[derive(Debug, Clone)]
struct PythonRandom {
    words: [u32; MT_N],
    index: usize,
}

impl PythonRandom {
    fn from_seed(seed: u64) -> Self {
        // CPython converts an integer seed to little-endian 32-bit limbs and
        // passes them to the reference MT19937 init_by_array routine.
        let low = seed as u32;
        let high = (seed >> 32) as u32;
        let key = if high == 0 {
            vec![low]
        } else {
            vec![low, high]
        };
        Self::from_key(&key)
    }

    fn from_key(key: &[u32]) -> Self {
        let mut rng = Self {
            words: [0; MT_N],
            index: MT_N,
        };
        rng.init_genrand(19_650_218);

        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = MT_N.max(key.len());
        while k > 0 {
            let previous = rng.words[i - 1];
            rng.words[i] = (rng.words[i] ^ ((previous ^ (previous >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                rng.words[0] = rng.words[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = MT_N - 1;
        while k > 0 {
            let previous = rng.words[i - 1];
            rng.words[i] = (rng.words[i]
                ^ ((previous ^ (previous >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                rng.words[0] = rng.words[MT_N - 1];
                i = 1;
            }
            k -= 1;
        }
        rng.words[0] = 0x8000_0000;
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.words[0] = seed;
        for i in 1..MT_N {
            let previous = self.words[i - 1];
            self.words[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = MT_N;
    }

    fn from_snapshot(snapshot: &PythonRandomSnapshot) -> Result<Self, TrainingError> {
        let words: [u32; MT_N] =
            snapshot.words.as_slice().try_into().map_err(|_| {
                TrainingError::InvalidSamplerState("MT19937 state must have 624 words")
            })?;
        if snapshot.index > MT_N {
            return Err(TrainingError::InvalidSamplerState(
                "MT19937 index exceeds 624",
            ));
        }
        Ok(Self {
            words,
            index: snapshot.index,
        })
    }

    fn snapshot(&self) -> PythonRandomSnapshot {
        PythonRandomSnapshot {
            words: self.words.to_vec(),
            index: self.index,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            for i in 0..(MT_N - MT_M) {
                let joined = (self.words[i] & UPPER_MASK) | (self.words[i + 1] & LOWER_MASK);
                self.words[i] = self.words[i + MT_M]
                    ^ (joined >> 1)
                    ^ if joined & 1 != 0 { MATRIX_A } else { 0 };
            }
            for i in (MT_N - MT_M)..(MT_N - 1) {
                let joined = (self.words[i] & UPPER_MASK) | (self.words[i + 1] & LOWER_MASK);
                self.words[i] = self.words[i - (MT_N - MT_M)]
                    ^ (joined >> 1)
                    ^ if joined & 1 != 0 { MATRIX_A } else { 0 };
            }
            let joined = (self.words[MT_N - 1] & UPPER_MASK) | (self.words[0] & LOWER_MASK);
            self.words[MT_N - 1] =
                self.words[MT_M - 1] ^ (joined >> 1) ^ if joined & 1 != 0 { MATRIX_A } else { 0 };
            self.index = 0;
        }

        let mut value = self.words[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn getrandbits(&mut self, bits: u32) -> u64 {
        debug_assert!((1..=64).contains(&bits));
        let mut value = 0u64;
        let mut shift = 0u32;
        let mut remaining = bits;
        while remaining > 0 {
            let take = remaining.min(32);
            let word = self.next_u32();
            let part = if take == 32 {
                word
            } else {
                word >> (32 - take)
            };
            value |= u64::from(part) << shift;
            shift += take;
            remaining -= take;
        }
        value
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bits = (usize::BITS - upper.leading_zeros()) as u32;
        loop {
            let value = self.getrandbits(bits);
            if value < upper as u64 {
                return value as usize;
            }
        }
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = self.randbelow(i + 1);
            values.swap(i, j);
        }
    }
}
