//! Exact one-excess CNOT recognizer using the source's bounded temporal
//! contraction. Counts use arbitrary-precision integers; resource exhaustion
//! never becomes a negative feature label.

use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::stage37::{Stage37Error, Stage37Result, recognize_stage37};
use crate::state::MatrixState;
use num_bigint::{BigInt, Sign};
use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const STAGE38_PRODUCER: &str = "stage38_one_excess_exact_v1";
pub const STAGE38_SOURCE_NOTES_SHA256: &str =
    "0815dc96c12052e04d26a04295b26f883c8936fb62d5b4c484b7d3841ee8292d";
pub const STAGE38_SOURCE_IMPLEMENTATION_SHA256: &str =
    "90f4aac082958111293665010a34332f28d5145e957ca2ae49551d199c4cd82e";

type Row = Vec<u64>;
type Histories = Vec<Vec<Row>>;
type Constraint = (usize, usize);

mod contraction;
use contraction::{count_histories, recover_histories};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stage38Error {
    AlphabetMismatch,
    InvalidBudget,
    SizeOverflow,
    InvalidHistory(&'static str),
    ContractionBudgetExceeded(String),
    ContractionInvariant(&'static str),
    Stage37(String),
    InvalidCertificate(String),
    ResultMismatch,
}

impl Display for Stage38Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlphabetMismatch => formatter.write_str("state/alphabet identity mismatch"),
            Self::InvalidBudget => formatter.write_str("positive contraction budget required"),
            Self::SizeOverflow => formatter.write_str("stage-38 size or index overflow"),
            Self::InvalidHistory(reason) => write!(formatter, "invalid row history: {reason}"),
            Self::ContractionBudgetExceeded(reason) => {
                write!(formatter, "exact contraction budget exceeded: {reason}")
            }
            Self::ContractionInvariant(reason) => {
                write!(formatter, "stage-38 contraction invariant failed: {reason}")
            }
            Self::Stage37(reason) => write!(formatter, "stage-37 exclusion failed: {reason}"),
            Self::InvalidCertificate(reason) => {
                write!(formatter, "stage-38 word certificate failed: {reason}")
            }
            Self::ResultMismatch => {
                formatter.write_str("stage-38 result differs from recomputation")
            }
        }
    }
}

impl Error for Stage38Error {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage38Exact {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub distance: u64,
    pub changed_rows: usize,
    pub templates: usize,
    pub one_excess_count: String,
    pub max_bag: usize,
    pub transport_free: bool,
    pub max_entries: usize,
    pub word: WordCertificate,
    pub source_notes_sha256: &'static str,
    pub source_implementation_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage38NoOneExcess {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub excluded_length: u64,
    pub changed_rows: usize,
    pub templates: usize,
    pub max_bag: usize,
    pub max_entries: usize,
    pub source_notes_sha256: &'static str,
    pub source_implementation_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage38Unsupported {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage38ResourceLimit {
    pub state: MatrixState,
    pub alphabet_id: String,
    pub max_entries: usize,
    pub changed_rows: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stage38Result {
    Exact(Stage38Exact),
    CertifiedNoSolutionWithinScope(Stage38NoOneExcess),
    Unsupported(Stage38Unsupported),
    ResourceLimit(Stage38ResourceLimit),
}

/// Recognizes the exact `r+1` CNOT family after stage 37 excludes length `r`.
pub fn recognize_stage38(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    max_entries: usize,
) -> Result<Stage38Result, Stage38Error> {
    if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
        return Err(Stage38Error::AlphabetMismatch);
    }
    if max_entries == 0 {
        return Err(Stage38Error::InvalidBudget);
    }
    if !is_complete_unit_alphabet(alphabet) {
        return Ok(unsupported(
            state,
            alphabet,
            "requires full unit all-to-all CNOT",
        ));
    }

    let zero_excess = match recognize_stage37(state, alphabet) {
        Ok(result) => result,
        Err(Stage37Error::AlphabetMismatch) => return Err(Stage38Error::AlphabetMismatch),
        Err(error @ Stage37Error::IndexOverflow) | Err(error @ Stage37Error::SizeOverflow) => {
            return Err(Stage38Error::Stage37(error.to_string()));
        }
        Err(error) => return Err(Stage38Error::Stage37(error.to_string())),
    };
    match zero_excess {
        Stage37Result::Exact(_) => {
            return Ok(unsupported(
                state,
                alphabet,
                "zero-excess labels belong to stage 37, not the one-excess family",
            ));
        }
        Stage37Result::Unsupported(_) => {
            return Ok(unsupported(
                state,
                alphabet,
                "zero-excess exclusion was not established",
            ));
        }
        Stage37Result::NoZeroExcess(_) => {}
    }

    one_excess(state, alphabet, max_entries)
}

/// Re-contracts exact results and replays their CNOT word.
pub fn verify_stage38_result(
    result: &Stage38Result,
    alphabet: &CnotAlphabet,
) -> Result<bool, Stage38Error> {
    let Stage38Result::Exact(exact) = result else {
        return Ok(false);
    };
    let reproduced = recognize_stage38(&exact.state, alphabet, exact.max_entries)?;
    if reproduced != *result {
        return Err(Stage38Error::ResultMismatch);
    }
    exact
        .word
        .verify(alphabet)
        .map_err(|error| Stage38Error::InvalidCertificate(error.to_string()))?;
    Ok(true)
}

fn one_excess(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    max_entries: usize,
) -> Result<Stage38Result, Stage38Error> {
    let n = usize::try_from(state.dimension()).map_err(|_| Stage38Error::SizeOverflow)?;
    let initial = (0..n)
        .map(|row| {
            state
                .row_words(u32::try_from(row).map_err(|_| Stage38Error::SizeOverflow)?)
                .map(<[u64]>::to_vec)
                .ok_or(Stage38Error::SizeOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let final_rows = (0..n)
        .map(|row| identity_row(n, state.words_per_row(), row))
        .collect::<Vec<_>>();
    let changed: Vec<usize> = (0..n)
        .filter(|row| initial[*row] != final_rows[*row])
        .collect();
    let changed_count = changed.len();
    let excluded_length = u64::try_from(changed_count)
        .ok()
        .and_then(|count| count.checked_add(1))
        .ok_or(Stage38Error::SizeOverflow)?;

    let mut total = BigInt::from(0u8);
    let mut templates = 0usize;
    let mut max_bag = 0usize;
    let mut chosen: Option<Histories> = None;

    for target in changed.iter().copied() {
        let mut endpoints = HashSet::new();
        for row in 0..n {
            if row == target {
                continue;
            }
            endpoints.insert(initial[row].clone());
            endpoints.insert(final_rows[row].clone());
        }

        let mut candidates = HashSet::new();
        for endpoint in &endpoints {
            let intermediate = xor_rows(&initial[target], endpoint);
            if intermediate != zero_row(state.words_per_row())
                && intermediate != initial[target]
                && intermediate != final_rows[target]
            {
                candidates.insert(intermediate);
            }
        }
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_by(numeric_row_cmp);

        for intermediate in candidates {
            if !endpoints.contains(&xor_rows(&intermediate, &final_rows[target])) {
                continue;
            }
            let histories: Histories = (0..n)
                .map(|row| {
                    if row == target {
                        vec![
                            initial[row].clone(),
                            intermediate.clone(),
                            final_rows[row].clone(),
                        ]
                    } else if initial[row] != final_rows[row] {
                        vec![initial[row].clone(), final_rows[row].clone()]
                    } else {
                        vec![initial[row].clone()]
                    }
                })
                .collect();
            templates = templates.checked_add(1).ok_or(Stage38Error::SizeOverflow)?;
            let audit = match count_histories(&histories, max_entries, &HashSet::new()) {
                Ok(audit) => audit,
                Err(Stage38Error::ContractionBudgetExceeded(reason)) => {
                    return Ok(resource_limit(
                        state,
                        alphabet,
                        max_entries,
                        changed_count,
                        reason,
                    ));
                }
                Err(error) => return Err(error),
            };
            max_bag = max_bag.max(audit.max_bag);
            total += &audit.count;
            if audit.count.sign() == Sign::Plus && chosen.is_none() {
                chosen = Some(histories);
            }
        }
    }

    if total.sign() == Sign::NoSign {
        return Ok(Stage38Result::CertifiedNoSolutionWithinScope(
            Stage38NoOneExcess {
                state: state.clone(),
                alphabet_id: alphabet.id().to_owned(),
                excluded_length,
                changed_rows: changed_count,
                templates,
                max_bag,
                max_entries,
                source_notes_sha256: STAGE38_SOURCE_NOTES_SHA256,
                source_implementation_sha256: STAGE38_SOURCE_IMPLEMENTATION_SHA256,
            },
        ));
    }

    let chosen = chosen.ok_or(Stage38Error::ContractionInvariant(
        "positive amplitude has no retained history template",
    ))?;
    let mut gates = match recover_histories(&chosen, max_entries) {
        Ok(gates) => gates,
        Err(Stage38Error::ContractionBudgetExceeded(reason)) => {
            return Ok(resource_limit(
                state,
                alphabet,
                max_entries,
                changed_count,
                reason,
            ));
        }
        Err(error) => return Err(error),
    };
    if gates.len()
        != changed_count
            .checked_add(1)
            .ok_or(Stage38Error::SizeOverflow)?
    {
        return Err(Stage38Error::ContractionInvariant(
            "recovered one-excess word has the wrong length",
        ));
    }

    let word = WordCertificate {
        start: state.clone(),
        end: MatrixState::identity(state.dimension(), alphabet.id())
            .map_err(|error| Stage38Error::InvalidCertificate(error.to_string()))?,
        gates: std::mem::take(&mut gates),
        physical_cost: excluded_length,
        producer: STAGE38_PRODUCER.into(),
        lineage_id: stage38_lineage(state)?,
    };
    word.verify(alphabet)
        .map_err(|error| Stage38Error::InvalidCertificate(error.to_string()))?;

    let transport_free =
        (0..n).all(|row| (0..n).all(|other| row == other || initial[row] != final_rows[other]));
    if transport_free && max_bag > 5 {
        return Err(Stage38Error::ContractionInvariant(
            "transport-free width bound was violated",
        ));
    }

    Ok(Stage38Result::Exact(Stage38Exact {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        distance: excluded_length,
        changed_rows: changed_count,
        templates,
        one_excess_count: total.to_string(),
        max_bag,
        transport_free,
        max_entries,
        word,
        source_notes_sha256: STAGE38_SOURCE_NOTES_SHA256,
        source_implementation_sha256: STAGE38_SOURCE_IMPLEMENTATION_SHA256,
    }))
}

fn is_complete_unit_alphabet(alphabet: &CnotAlphabet) -> bool {
    let Ok(expected) = CnotAlphabet::all_to_all_unit(alphabet.dimension()) else {
        return false;
    };
    alphabet.id() == expected.id()
        && alphabet.gates() == expected.gates()
        && alphabet
            .gates()
            .iter()
            .enumerate()
            .all(|(index, _)| alphabet.cost_at(index) == Some(1))
}

fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> Stage38Result {
    Stage38Result::Unsupported(Stage38Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    })
}

fn resource_limit(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    max_entries: usize,
    changed_rows: usize,
    reason: String,
) -> Stage38Result {
    Stage38Result::ResourceLimit(Stage38ResourceLimit {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        max_entries,
        changed_rows,
        reason,
    })
}

pub(super) fn xor_rows(left: &Row, right: &Row) -> Row {
    left.iter()
        .zip(right)
        .map(|(left, right)| left ^ right)
        .collect()
}

fn zero_row(words: usize) -> Row {
    vec![0; words]
}

fn identity_row(n: usize, words: usize, row: usize) -> Row {
    let mut value = vec![0; words];
    value[row / 64] = 1u64 << (row % 64);
    debug_assert!(row < n);
    value
}

fn numeric_row_cmp(left: &Row, right: &Row) -> std::cmp::Ordering {
    left.iter().rev().cmp(right.iter().rev())
}

fn stage38_lineage(state: &MatrixState) -> Result<String, Stage38Error> {
    let n = usize::try_from(state.dimension()).map_err(|_| Stage38Error::SizeOverflow)?;
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let words = state
            .row_words(u32::try_from(row).map_err(|_| Stage38Error::SizeOverflow)?)
            .ok_or(Stage38Error::SizeOverflow)?;
        let mut value = BigInt::from(0u8);
        for word in words.iter().rev() {
            value <<= 64usize;
            value += BigInt::from(*word);
        }
        rows.push(value.to_string());
    }
    let tuple = if rows.len() == 1 {
        format!("({},)", rows[0])
    } else {
        format!("({})", rows.join(", "))
    };
    Ok(format!("stage38:{}:{tuple}", state.dimension()))
}
