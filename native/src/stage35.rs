//! Exact stage-35 two-changed-row recognizer from the own-source v13 package.
//!
//! This is a bounded exact oracle, not a general distance estimator. It only
//! returns an exact label when the full unit-cost CNOT alphabet and the
//! at-most-two-changed-row hypotheses hold and the constructed word matches
//! the independent global lower bound.

use crate::alphabet::{CertificateError, CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub const STAGE35_THEOREM_SHA256: &str =
    "1b8ef924af6c68af51dd67e0be98148ad7f8e8c272d379ad44943b7baf3fade5";
pub const STAGE35_PRODUCER: &str = "stage35_pair_exact_v1";

const IDENTITY_FRAME: [u8; 2] = [1, 2];
const FRAME_IDENTITY: &[(usize, usize)] = &[];
const FRAME_13: &[(usize, usize)] = &[(0, 1)];
const FRAME_32: &[(usize, usize)] = &[(1, 0)];
const FRAME_23: &[(usize, usize)] = &[(0, 1), (1, 0)];
const FRAME_31: &[(usize, usize)] = &[(1, 0), (0, 1)];
const FRAME_21: &[(usize, usize)] = &[(0, 1), (1, 0), (0, 1)];
const FRAME_IDENTITY_WITH_SHARED: &[(usize, usize)] = &[(0, 1), (0, 1)];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage35Error {
    AlphabetMismatch,
    SingularActiveBlock,
    MissingFrame([u8; 2]),
    IndexOverflow,
    CostOverflow,
    FormulaWordLengthMismatch { formula: u64, word: usize },
    TheoremMismatch { distance: u64, lower_bound: u64 },
    InvalidCertificate(CertificateError),
    ResultMismatch,
}

impl Display for Stage35Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphabetMismatch => formatter.write_str("state and CNOT alphabet differ"),
            Self::SingularActiveBlock => formatter.write_str("active 2x2 frame is singular"),
            Self::MissingFrame(frame) => write!(formatter, "no stage-35 route for frame {frame:?}"),
            Self::IndexOverflow => formatter.write_str("stage-35 wire index exceeds u32"),
            Self::CostOverflow => formatter.write_str("stage-35 cost exceeds u64"),
            Self::FormulaWordLengthMismatch { formula, word } => write!(
                formatter,
                "stage-35 formula gives {formula} gates but constructed word has {word}"
            ),
            Self::TheoremMismatch {
                distance,
                lower_bound,
            } => write!(
                formatter,
                "stage-35 word cost {distance} differs from certified lower bound {lower_bound}"
            ),
            Self::InvalidCertificate(error) => Display::fmt(error, formatter),
            Self::ResultMismatch => {
                formatter.write_str("stage-35 result differs from re-recognition")
            }
        }
    }
}

impl Error for Stage35Error {}

impl From<CertificateError> for Stage35Error {
    fn from(error: CertificateError) -> Self {
        Self::InvalidCertificate(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage35Branch {
    NoFrameMotion,
    NonemptyFrameMotion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage35LowerBoundKind {
    ConnectivityPlusDiagonalZeros,
    TreeObstructionTwoSharedSources,
    Connectivity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage35Exact {
    pub alphabet_id: String,
    pub distance: u64,
    pub word: WordCertificate,
    pub theorem_sha256: &'static str,
    pub changed_rows: Vec<u32>,
    pub active_rows: [u32; 2],
    /// The two active rows restricted to the active columns, encoded as 1/2 bits.
    pub frame_rows: [u8; 2],
    pub nonzero_external_columns: usize,
    pub shared_external_columns: usize,
    pub branch: Stage35Branch,
    pub lower_bound_kind: Stage35LowerBoundKind,
    pub connectivity: usize,
    pub zero_diagonal: usize,
    pub shared_sources: Vec<u32>,
    pub lower_bound: u64,
    pub encoder_gates: Vec<CnotGate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage35Unsupported {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stage35Result {
    Exact(Stage35Exact),
    Unsupported(Stage35Unsupported),
}

/// Recognize the theorem's exact family and return a replayable shortest word.
/// Unsupported inputs have no distance label and are never encoded as zero.
pub fn recognize_stage35(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<Stage35Result, Stage35Error> {
    if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
        return Err(Stage35Error::AlphabetMismatch);
    }
    if !is_full_unit_alphabet(alphabet)? {
        return Ok(unsupported(
            state,
            alphabet,
            "requires full unit CNOT alphabet",
        ));
    }
    let n = usize::try_from(state.dimension()).map_err(|_| Stage35Error::IndexOverflow)?;
    if n < 2 {
        return Ok(unsupported(state, alphabet, "requires at least two wires"));
    }

    let changed_rows: Vec<usize> = (0..n).filter(|row| !is_identity_row(state, *row)).collect();
    if changed_rows.len() > 2 {
        return Ok(unsupported(state, alphabet, "more than two changed rows"));
    }

    let mut active = changed_rows.clone();
    for row in 0..n {
        if active.len() == 2 {
            break;
        }
        if !active.contains(&row) {
            active.push(row);
        }
    }
    let active_rows = [active[0], active[1]];
    let frame_rows = [
        restricted_row(state, active[0], active_rows),
        restricted_row(state, active[1], active_rows),
    ];
    let frame_inverse = inverse_2x2(frame_rows)?;
    let external: Vec<usize> = (0..n).filter(|row| !active.contains(row)).collect();

    let columns: Vec<(usize, u8)> = external
        .iter()
        .copied()
        .map(|column| {
            let value = u8::from(bit(state, active[0], column))
                | (u8::from(bit(state, active[1], column)) << 1);
            (column, value)
        })
        .collect();
    let transformed: Vec<(usize, u8)> = columns
        .iter()
        .map(|(column, value)| (*column, multiply_2x2(frame_inverse, *value)))
        .collect();
    let nonzero_external_columns = transformed.iter().filter(|(_, value)| *value != 0).count();
    let shared: Vec<usize> = transformed
        .iter()
        .filter_map(|(column, value)| (*value == 3).then_some(*column))
        .collect();
    let shared_external_columns = shared.len();
    let identity_frame = frame_rows == IDENTITY_FRAME;

    let (branch, frame, formula_cost) = if identity_frame && shared.len() <= 2 {
        let cost = nonzero_external_columns
            .checked_add(shared.len())
            .ok_or(Stage35Error::CostOverflow)?;
        (Stage35Branch::NoFrameMotion, FRAME_IDENTITY, cost)
    } else {
        let frame = if identity_frame {
            FRAME_IDENTITY_WITH_SHARED
        } else {
            frame_for(frame_rows).ok_or(Stage35Error::MissingFrame(frame_rows))?
        };
        let cost = nonzero_external_columns
            .checked_add(frame.len())
            .ok_or(Stage35Error::CostOverflow)?;
        (Stage35Branch::NonemptyFrameMotion, frame, cost)
    };

    let mut encoder_gates = Vec::with_capacity(formula_cost);
    match branch {
        Stage35Branch::NoFrameMotion => {
            for (column, value) in columns {
                if value & 1 != 0 {
                    encoder_gates.push(gate(column, active[0])?);
                }
                if value & 2 != 0 {
                    encoder_gates.push(gate(column, active[1])?);
                }
            }
        }
        Stage35Branch::NonemptyFrameMotion => {
            for (column, value) in &transformed {
                if *value == 1 || *value == 2 {
                    let target = if *value == 1 { active[0] } else { active[1] };
                    encoder_gates.push(gate(*column, target)?);
                }
            }
            let (control, target) = frame[0];
            encoder_gates.push(gate(active[control], active[target])?);

            let mut first_frame = IDENTITY_FRAME;
            first_frame[target] ^= first_frame[control];
            let first_inverse = inverse_2x2(first_frame)?;
            let receiver = (0..2)
                .find(|row| multiply_2x2(first_inverse, 1u8 << row) == 3)
                .ok_or(Stage35Error::MissingFrame(first_frame))?;
            for column in &shared {
                encoder_gates.push(gate(*column, active[receiver])?);
            }
            for (control, target) in frame.iter().copied().skip(1) {
                encoder_gates.push(gate(active[control], active[target])?);
            }
        }
    }

    if encoder_gates.len() != formula_cost {
        return Err(Stage35Error::FormulaWordLengthMismatch {
            formula: u64::try_from(formula_cost).map_err(|_| Stage35Error::CostOverflow)?,
            word: encoder_gates.len(),
        });
    }
    let distance = u64::try_from(formula_cost).map_err(|_| Stage35Error::CostOverflow)?;
    let identity = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|_| Stage35Error::AlphabetMismatch)?;
    let word = WordCertificate {
        start: state.clone(),
        end: identity,
        gates: encoder_gates.iter().rev().copied().collect(),
        physical_cost: distance,
        producer: STAGE35_PRODUCER.into(),
        lineage_id: format!("stage35:{}:{:?}", state.dimension(), state.packed_rows()),
    };
    word.verify(alphabet)?;

    let connectivity = state.dimension() as usize - support_components(state);
    let zero_diagonal = (0..n).filter(|index| !bit(state, *index, *index)).count();
    let (lower_bound, lower_bound_kind) = if !identity_frame {
        (
            connectivity
                .checked_add(zero_diagonal)
                .ok_or(Stage35Error::CostOverflow)?,
            Stage35LowerBoundKind::ConnectivityPlusDiagonalZeros,
        )
    } else if shared.len() >= 2 {
        (
            connectivity
                .checked_add(1)
                .ok_or(Stage35Error::CostOverflow)?,
            Stage35LowerBoundKind::TreeObstructionTwoSharedSources,
        )
    } else {
        (connectivity, Stage35LowerBoundKind::Connectivity)
    };
    let lower_bound = u64::try_from(lower_bound).map_err(|_| Stage35Error::CostOverflow)?;
    if lower_bound != distance {
        return Err(Stage35Error::TheoremMismatch {
            distance,
            lower_bound,
        });
    }

    let changed_rows = changed_rows
        .into_iter()
        .map(|row| u32::try_from(row).map_err(|_| Stage35Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let active_rows = [
        u32::try_from(active_rows[0]).map_err(|_| Stage35Error::IndexOverflow)?,
        u32::try_from(active_rows[1]).map_err(|_| Stage35Error::IndexOverflow)?,
    ];
    let shared_sources = shared
        .into_iter()
        .take(2)
        .map(|row| u32::try_from(row).map_err(|_| Stage35Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Stage35Result::Exact(Stage35Exact {
        alphabet_id: alphabet.id().to_owned(),
        distance,
        word,
        theorem_sha256: STAGE35_THEOREM_SHA256,
        changed_rows,
        active_rows,
        frame_rows,
        nonzero_external_columns,
        shared_external_columns,
        branch,
        lower_bound_kind,
        connectivity,
        zero_diagonal,
        shared_sources,
        lower_bound,
        encoder_gates,
    }))
}

/// Recomputes the recognition decision and, for exact results, replays its word.
pub fn verify_stage35_result(
    result: &Stage35Result,
    alphabet: &CnotAlphabet,
) -> Result<bool, Stage35Error> {
    let state = match result {
        Stage35Result::Exact(exact) => &exact.word.start,
        Stage35Result::Unsupported(unsupported) => &unsupported.state,
    };
    let reproduced = recognize_stage35(state, alphabet)?;
    if &reproduced != result {
        return Err(Stage35Error::ResultMismatch);
    }
    if let Stage35Result::Exact(exact) = result {
        exact.word.verify(alphabet)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> Stage35Result {
    Stage35Result::Unsupported(Stage35Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    })
}

fn is_full_unit_alphabet(alphabet: &CnotAlphabet) -> Result<bool, Stage35Error> {
    let n = usize::try_from(alphabet.dimension()).map_err(|_| Stage35Error::IndexOverflow)?;
    let Some(action_count) = n.checked_mul(n.saturating_sub(1)) else {
        return Ok(false);
    };
    if alphabet.gates().len() != action_count {
        return Ok(false);
    }
    if alphabet
        .gates()
        .iter()
        .enumerate()
        .any(|(index, _)| alphabet.cost_at(index) != Some(1))
    {
        return Ok(false);
    }
    Ok(true)
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

fn bit(state: &MatrixState, row: usize, column: usize) -> bool {
    state.row_words(row as u32).is_some_and(|words| {
        words
            .get(column / 64)
            .is_some_and(|word| word & (1u64 << (column % 64)) != 0)
    })
}

fn restricted_row(state: &MatrixState, row: usize, active_rows: [usize; 2]) -> u8 {
    u8::from(bit(state, row, active_rows[0])) | (u8::from(bit(state, row, active_rows[1])) << 1)
}

fn multiply_2x2(matrix_rows: [u8; 2], vector: u8) -> u8 {
    ((matrix_rows[0] & vector).count_ones() as u8 & 1)
        | (((matrix_rows[1] & vector).count_ones() as u8 & 1) << 1)
}

fn inverse_2x2(rows: [u8; 2]) -> Result<[u8; 2], Stage35Error> {
    let a = rows[0] & 1;
    let x = (rows[0] >> 1) & 1;
    let y = rows[1] & 1;
    let d = (rows[1] >> 1) & 1;
    if (a & d) ^ (x & y) != 1 {
        return Err(Stage35Error::SingularActiveBlock);
    }
    Ok([d | (x << 1), y | (a << 1)])
}

fn frame_for(rows: [u8; 2]) -> Option<&'static [(usize, usize)]> {
    match rows {
        [1, 2] => Some(FRAME_IDENTITY),
        [1, 3] => Some(FRAME_13),
        [3, 2] => Some(FRAME_32),
        [2, 3] => Some(FRAME_23),
        [3, 1] => Some(FRAME_31),
        [2, 1] => Some(FRAME_21),
        _ => None,
    }
}

fn gate(control: usize, target: usize) -> Result<CnotGate, Stage35Error> {
    Ok(CnotGate {
        control: u32::try_from(control).map_err(|_| Stage35Error::IndexOverflow)?,
        target: u32::try_from(target).map_err(|_| Stage35Error::IndexOverflow)?,
    })
}

fn support_components(state: &MatrixState) -> usize {
    let n = state.dimension() as usize;
    let mut parent: Vec<usize> = (0..n).collect();
    for row in 0..n {
        for column in 0..n {
            if row != column && bit(state, row, column) {
                union(&mut parent, row, column);
            }
        }
    }
    let mut roots = std::collections::HashSet::with_capacity(n);
    for row in 0..n {
        roots.insert(find(&mut parent, row));
    }
    roots.len()
}

fn find(parent: &mut [usize], mut value: usize) -> usize {
    while parent[value] != value {
        let grandparent = parent[parent[value]];
        parent[value] = grandparent;
        value = grandparent;
    }
    value
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        parent[left_root] = right_root;
    }
}
