//! Exact recognizer for the zero-excess CNOT sector `A -> I`.
//!
//! For a complete unit-cost CNOT alphabet, every changed target row must be
//! updated at least once. This recognizer checks whether each changed row can
//! reach identity with exactly one update, orders those updates by a DAG, and
//! returns a replayable word. Failure excludes only the changed-row lower
//! bound; it does not estimate the true distance. It is a solver primitive,
//! not an F07 packet provider: F07's native group/incidence contract remains
//! unconfigured.

use crate::alphabet::{CertificateError, CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const STAGE37_PRODUCER: &str = "stage37:zero_excess";
pub const STAGE37_SOURCE_NOTES_SHA256: &str =
    "144cfd8d91663581fee9edb79718d5fb93969dcda50a42e8939965024834c6f9";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stage37Error {
    AlphabetMismatch,
    IndexOverflow,
    SizeOverflow,
    AcyclicityContradiction,
    InvalidCertificate(CertificateError),
    ResultMismatch,
}

impl Display for Stage37Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlphabetMismatch => {
                formatter.write_str("state dimension or alphabet identity does not match")
            }
            Self::IndexOverflow => formatter.write_str("stage-37 row index does not fit u32"),
            Self::SizeOverflow => formatter.write_str("stage-37 lower bound does not fit u64"),
            Self::AcyclicityContradiction => {
                formatter.write_str("stage-37 acyclicity theorem was contradicted")
            }
            Self::InvalidCertificate(error) => Display::fmt(error, formatter),
            Self::ResultMismatch => {
                formatter.write_str("stage-37 result differs from fresh recognition")
            }
        }
    }
}

impl Error for Stage37Error {}

impl From<CertificateError> for Stage37Error {
    fn from(error: CertificateError) -> Self {
        Self::InvalidCertificate(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage37ParentMode {
    InitialRow,
    FinalRow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stage37Parent {
    pub target_row: u32,
    pub control_row: u32,
    pub mode: Stage37ParentMode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage37Exact {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub distance: u64,
    pub lower_bound: u64,
    pub changed_rows: Vec<u32>,
    pub update_order: Vec<u32>,
    pub parents: Vec<Stage37Parent>,
    pub word: WordCertificate,
    pub source_notes_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage37NoZeroExcess {
    pub state: MatrixState,
    pub alphabet_id: String,
    /// The only length excluded by this result; longer distances are unknown.
    pub excluded_length: u64,
    pub missing_rows: Vec<u32>,
    pub source_notes_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage37Unsupported {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stage37Result {
    Exact(Stage37Exact),
    NoZeroExcess(Stage37NoZeroExcess),
    Unsupported(Stage37Unsupported),
}

/// Recognizes words from `state` to identity that attain the changed-row bound.
pub fn recognize_stage37(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<Stage37Result, Stage37Error> {
    if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
        return Err(Stage37Error::AlphabetMismatch);
    }
    if !is_complete_unit_alphabet(alphabet)? {
        return Ok(unsupported(
            state,
            alphabet,
            "requires complete unit-cost all-to-all CNOT alphabet",
        ));
    }

    let dimension = usize::try_from(state.dimension()).map_err(|_| Stage37Error::IndexOverflow)?;
    let identity = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|_| Stage37Error::AlphabetMismatch)?;
    let changed: Vec<usize> = (0..dimension)
        .filter(|row| !is_identity_row(state, *row))
        .collect();
    let mut changed_set = vec![false; dimension];
    for row in changed.iter().copied() {
        changed_set[row] = true;
    }

    let mut initial_rows = HashMap::with_capacity(dimension);
    for row in 0..dimension {
        initial_rows.insert(state.row_words(row as u32).unwrap().to_vec(), row);
    }
    let mut final_rows = HashMap::with_capacity(dimension);
    for row in 0..dimension {
        let mut value = vec![0u64; state.words_per_row()];
        value[row / 64] = 1u64 << (row % 64);
        final_rows.insert(value, row);
    }

    let mut parents = vec![None; dimension];
    let mut edges = vec![BTreeSet::new(); dimension];
    let mut indegree = vec![0usize; dimension];
    let mut missing = Vec::new();

    for target in changed.iter().copied() {
        let mut difference = state.row_words(target as u32).unwrap().to_vec();
        difference[target / 64] ^= 1u64 << (target % 64);

        let initial_parent = initial_rows
            .get(&difference)
            .copied()
            .filter(|parent| *parent != target);
        let parent = if let Some(control) = initial_parent {
            Some((control, Stage37ParentMode::InitialRow))
        } else {
            final_rows
                .get(&difference)
                .copied()
                .filter(|parent| *parent != target)
                .map(|control| (control, Stage37ParentMode::FinalRow))
        };

        let Some((control, mode)) = parent else {
            missing.push(target);
            continue;
        };
        parents[target] = Some((control, mode));

        if changed_set[control] {
            let (source, destination) = match mode {
                Stage37ParentMode::InitialRow => (target, control),
                Stage37ParentMode::FinalRow => (control, target),
            };
            if edges[source].insert(destination) {
                indegree[destination] += 1;
            }
        }
    }

    let lower_bound = u64::try_from(changed.len()).map_err(|_| Stage37Error::SizeOverflow)?;
    if !missing.is_empty() {
        let missing_rows = missing
            .into_iter()
            .map(|row| u32::try_from(row).map_err(|_| Stage37Error::IndexOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Stage37Result::NoZeroExcess(Stage37NoZeroExcess {
            state: state.clone(),
            alphabet_id: alphabet.id().to_owned(),
            excluded_length: lower_bound,
            missing_rows,
            source_notes_sha256: STAGE37_SOURCE_NOTES_SHA256,
        }));
    }

    let mut queue = VecDeque::new();
    for row in changed.iter().copied() {
        if indegree[row] == 0 {
            queue.push_back(row);
        }
    }
    let mut order = Vec::with_capacity(changed.len());
    while let Some(row) = queue.pop_front() {
        order.push(row);
        for dependent in edges[row].iter().copied() {
            indegree[dependent] -= 1;
            if indegree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }
    if order.len() != changed.len() {
        return Err(Stage37Error::AcyclicityContradiction);
    }

    let gate_list = order
        .iter()
        .copied()
        .map(|target| {
            let (control, _) = parents[target].ok_or(Stage37Error::AcyclicityContradiction)?;
            Ok(CnotGate {
                control: u32::try_from(control).map_err(|_| Stage37Error::IndexOverflow)?,
                target: u32::try_from(target).map_err(|_| Stage37Error::IndexOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, Stage37Error>>()?;

    let word = WordCertificate {
        start: state.clone(),
        end: identity,
        gates: gate_list,
        physical_cost: lower_bound,
        producer: STAGE37_PRODUCER.into(),
        lineage_id: lineage_id(state),
    };
    word.verify(alphabet)?;

    let changed_rows = changed
        .iter()
        .copied()
        .map(|row| u32::try_from(row).map_err(|_| Stage37Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let update_order = order
        .iter()
        .copied()
        .map(|row| u32::try_from(row).map_err(|_| Stage37Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let ordered_parents = order
        .iter()
        .copied()
        .map(|target| {
            let (control, mode) = parents[target].ok_or(Stage37Error::AcyclicityContradiction)?;
            Ok(Stage37Parent {
                target_row: u32::try_from(target).map_err(|_| Stage37Error::IndexOverflow)?,
                control_row: u32::try_from(control).map_err(|_| Stage37Error::IndexOverflow)?,
                mode,
            })
        })
        .collect::<Result<Vec<_>, Stage37Error>>()?;

    Ok(Stage37Result::Exact(Stage37Exact {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        distance: lower_bound,
        lower_bound,
        changed_rows,
        update_order,
        parents: ordered_parents,
        word,
        source_notes_sha256: STAGE37_SOURCE_NOTES_SHA256,
    }))
}

/// Re-recognizes the exact sector and replays its word certificate.
pub fn verify_stage37_result(
    result: &Stage37Result,
    alphabet: &CnotAlphabet,
) -> Result<bool, Stage37Error> {
    let Stage37Result::Exact(exact) = result else {
        return Ok(false);
    };
    let reproduced = recognize_stage37(&exact.state, alphabet)?;
    if reproduced != *result {
        return Err(Stage37Error::ResultMismatch);
    }
    exact.word.verify(alphabet)?;
    Ok(true)
}

fn is_complete_unit_alphabet(alphabet: &CnotAlphabet) -> Result<bool, Stage37Error> {
    let n = usize::try_from(alphabet.dimension()).map_err(|_| Stage37Error::IndexOverflow)?;
    let action_count = n
        .checked_mul(n.saturating_sub(1))
        .ok_or(Stage37Error::SizeOverflow)?;
    if alphabet.gates().len() != action_count {
        return Ok(false);
    }
    Ok(alphabet
        .gates()
        .iter()
        .enumerate()
        .all(|(index, gate)| alphabet.cost_at(index) == Some(1) && gate.control != gate.target))
}

fn is_identity_row(state: &MatrixState, row: usize) -> bool {
    state.row_words(row as u32).is_some_and(|words| {
        words.iter().enumerate().all(|(word, value)| {
            *value
                == if word == row / 64 {
                    1u64 << (row % 64)
                } else {
                    0
                }
        })
    })
}

fn lineage_id(state: &MatrixState) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103-stage37-lineage-v1");
    digest.update(state.dimension().to_le_bytes());
    for word in state.packed_rows() {
        digest.update(word.to_le_bytes());
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("stage37:{}:{hex}", state.dimension())
}

fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> Stage37Result {
    Stage37Result::Unsupported(Stage37Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    })
}
