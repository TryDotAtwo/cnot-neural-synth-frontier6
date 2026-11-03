use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use super::TrainingDataError;

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
