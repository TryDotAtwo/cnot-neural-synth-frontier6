//! Verified training records and exact-distance catalogs.
//!
//! Exact labels are admitted only from a finite, alphabet-closed catalog with
//! replayable reductions and a globally 1-Lipschitz distance potential. This
//! is stronger than treating one short word as a shortest-path proof.

use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingDataError(pub String);

impl Display for TrainingDataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TrainingDataError {}

/// Port of the source `cheap_bounds` function: nonzero rows/columns and GF(2)
/// rank of `M xor I`, combined by their maximum.
pub fn elementary_lower_bound(state: &MatrixState) -> Result<u64, TrainingDataError> {
    let n = usize::try_from(state.dimension())
        .map_err(|_| TrainingDataError("matrix dimension exceeds host index range".into()))?;
    let words_per_row = state.words_per_row();
    let mut residual = Vec::new();
    residual
        .try_reserve_exact(n)
        .map_err(|_| TrainingDataError("could not allocate lower-bound rows".into()))?;

    let mut changed_rows = 0usize;
    let mut changed_columns = vec![false; n];
    for row_index in 0..n {
        let start = row_index
            .checked_mul(words_per_row)
            .ok_or_else(|| TrainingDataError("matrix row offset overflow".into()))?;
        let mut row = state.packed_rows()[start..start + words_per_row].to_vec();
        row[row_index / 64] ^= 1u64 << (row_index % 64);
        changed_rows += usize::from(row.iter().any(|word| *word != 0));
        for (column, changed) in changed_columns.iter_mut().enumerate() {
            if ((row[column / 64] >> (column % 64)) & 1) != 0 {
                *changed = true;
            }
        }
        residual.push(row);
    }
    let rank = gf2_rank_packed(&mut residual, n);
    let changed_columns = changed_columns
        .into_iter()
        .filter(|changed| *changed)
        .count();
    Ok(changed_rows.max(changed_columns).max(rank) as u64)
}

fn gf2_rank_packed(rows: &mut [Vec<u64>], n: usize) -> usize {
    let mut pivots: Vec<Option<Vec<u64>>> = vec![None; n];
    let mut rank = 0usize;
    for row in rows {
        for column in (0..n).rev() {
            if ((row[column / 64] >> (column % 64)) & 1) == 0 {
                continue;
            }
            if let Some(pivot) = &pivots[column] {
                for (word, pivot_word) in row.iter_mut().zip(pivot) {
                    *word ^= pivot_word;
                }
            } else {
                pivots[column] = Some(row.clone());
                rank += 1;
                break;
            }
        }
    }
    rank
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingSplit {
    Train,
    Dev,
    Test,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactCatalogEntry {
    pub distance: u64,
    pub certificate: WordCertificate,
}

/// An exact-distance table whose proof obligations were checked at creation.
#[derive(Clone, Debug)]
pub struct ExactCatalog {
    alphabet_id: String,
    distances: HashMap<MatrixState, u64>,
    entries: HashMap<MatrixState, ExactCatalogEntry>,
    sha256: String,
}

impl ExactCatalog {
    /// Verify every state, reduction word, outgoing edge, and Lipschitz step
    /// before retaining the catalog as a source of exact labels.
    pub fn new(
        alphabet: &CnotAlphabet,
        entries: HashMap<MatrixState, ExactCatalogEntry>,
    ) -> Result<Self, TrainingDataError> {
        if entries.is_empty() {
            return Err(TrainingDataError("exact catalog is empty".into()));
        }
        let identity = MatrixState::identity(alphabet.dimension(), alphabet.id())
            .map_err(|error| TrainingDataError(error.to_string()))?;
        if entries.get(&identity).map(|entry| entry.distance) != Some(0) {
            return Err(TrainingDataError(
                "exact catalog must assign distance zero to identity".into(),
            ));
        }

        for (state, entry) in &entries {
            if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
                return Err(TrainingDataError(
                    "exact catalog contains a state from another alphabet".into(),
                ));
            }
            if entry.certificate.start != *state
                || entry.certificate.end != identity
                || entry.certificate.physical_cost != entry.distance
            {
                return Err(TrainingDataError(
                    "exact catalog reduction certificate does not match its state and distance"
                        .into(),
                ));
            }
            entry
                .certificate
                .verify(alphabet)
                .map_err(|error| TrainingDataError(error.to_string()))?;

            for (gate_index, gate) in alphabet.gates().iter().copied().enumerate() {
                let cost =
                    u64::from(alphabet.cost_at(gate_index).ok_or_else(|| {
                        TrainingDataError("alphabet gate cost is missing".into())
                    })?);
                let mut neighbor = state.clone();
                neighbor
                    .apply_cnot(gate)
                    .map_err(|error| TrainingDataError(error.to_string()))?;
                let neighbor_distance = entries
                    .get(&neighbor)
                    .ok_or_else(|| {
                        TrainingDataError("exact catalog is not closed under its alphabet".into())
                    })?
                    .distance;
                if entry.distance.abs_diff(neighbor_distance) > cost {
                    return Err(TrainingDataError(
                        "exact catalog distances violate the alphabet cost Lipschitz bound".into(),
                    ));
                }
            }
        }

        // A replayable path gives d(state) <= path_cost; the closed
        // 1-Lipschitz potential gives d(state) >= every path_cost to identity.
        // Equality of both values proves every stored distance exact.
        let distances: HashMap<_, _> = entries
            .iter()
            .map(|(state, entry)| (state.clone(), entry.distance))
            .collect();
        let sha256 = catalog_digest(alphabet.id(), &entries);
        Ok(Self {
            alphabet_id: alphabet.id().to_owned(),
            distances,
            entries,
            sha256,
        })
    }

    pub fn alphabet_id(&self) -> &str {
        &self.alphabet_id
    }

    pub fn exact_distance(&self, state: &MatrixState) -> Option<u64> {
        self.distances.get(state).copied()
    }

    pub fn state_count(&self) -> usize {
        self.distances.len()
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn entries(&self) -> &HashMap<MatrixState, ExactCatalogEntry> {
        &self.entries
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingRecord {
    pub state: MatrixState,
    pub split: TrainingSplit,
    pub verified_words: Vec<WordCertificate>,
    pub exact_distance: Option<u64>,
    pub lineage_id: String,
}

impl TrainingRecord {
    /// Validate training-only provenance and return its best verified upper bound.
    pub fn validate_for_training(
        &self,
        alphabet: &CnotAlphabet,
        exact_catalog: Option<&ExactCatalog>,
    ) -> Result<u64, TrainingDataError> {
        if self.split != TrainingSplit::Train {
            return Err(TrainingDataError(
                "training record must belong to the train split".into(),
            ));
        }
        if self.lineage_id.trim().is_empty() {
            return Err(TrainingDataError(
                "training record lineage identity is required".into(),
            ));
        }
        if self.state.dimension() != alphabet.dimension()
            || self.state.alphabet_id() != alphabet.id()
        {
            return Err(TrainingDataError(
                "training record state does not match its alphabet".into(),
            ));
        }
        if self.verified_words.is_empty() {
            return Err(TrainingDataError(
                "training record has no verified upper-bound word".into(),
            ));
        }

        let identity = MatrixState::identity(alphabet.dimension(), alphabet.id())
            .map_err(|error| TrainingDataError(error.to_string()))?;
        let mut upper_bound = u64::MAX;
        for certificate in &self.verified_words {
            if certificate.start != self.state || certificate.end != identity {
                return Err(TrainingDataError(
                    "upper-bound word is not a reduction of the training state".into(),
                ));
            }
            certificate
                .verify(alphabet)
                .map_err(|error| TrainingDataError(error.to_string()))?;
            upper_bound = upper_bound.min(certificate.physical_cost);
        }
        let lower_bound = elementary_lower_bound(&self.state)?;
        if lower_bound > upper_bound {
            return Err(TrainingDataError(
                "verified word upper bound is below the elementary lower bound".into(),
            ));
        }

        if let Some(exact_distance) = self.exact_distance {
            if exact_distance < lower_bound || exact_distance > upper_bound {
                return Err(TrainingDataError(
                    "exact label lies outside its verified lower/upper bounds".into(),
                ));
            }
            let proved_by_bounds = lower_bound == exact_distance && upper_bound == exact_distance;
            if !proved_by_bounds {
                let catalog = exact_catalog.ok_or_else(|| {
                    TrainingDataError(
                        "exact label needs a bound-equality proof or verified exact catalog".into(),
                    )
                })?;
                if catalog.alphabet_id() != alphabet.id()
                    || catalog.exact_distance(&self.state) != Some(exact_distance)
                {
                    return Err(TrainingDataError(
                        "exact label differs from the verified exact catalog".into(),
                    ));
                }
            } else if exact_catalog.is_some_and(|catalog| {
                catalog.alphabet_id() != alphabet.id()
                    || catalog.exact_distance(&self.state) != Some(exact_distance)
            }) {
                return Err(TrainingDataError(
                    "exact label differs from the verified exact catalog".into(),
                ));
            }
        }
        Ok(upper_bound)
    }
}

fn catalog_digest(alphabet_id: &str, entries: &HashMap<MatrixState, ExactCatalogEntry>) -> String {
    let mut rows: Vec<Vec<u8>> = entries
        .iter()
        .map(|(state, entry)| {
            let mut row = Vec::new();
            row.extend_from_slice(&state.dimension().to_le_bytes());
            append_bytes(&mut row, state.alphabet_id().as_bytes());
            row.extend_from_slice(&(state.packed_rows().len() as u64).to_le_bytes());
            for word in state.packed_rows() {
                row.extend_from_slice(&word.to_le_bytes());
            }
            row.extend_from_slice(&entry.distance.to_le_bytes());
            row
        })
        .collect();
    rows.sort();

    let mut digest = Sha256::new();
    append_digest_bytes(&mut digest, alphabet_id.as_bytes());
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        append_digest_bytes(&mut digest, &row);
    }
    let bytes: [u8; 32] = digest.finalize().into();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn append_bytes(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

fn append_digest_bytes(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}
