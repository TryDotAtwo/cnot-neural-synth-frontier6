use crate::alphabet::WordCertificate;
use crate::state::MatrixState;
use std::error::Error;
use std::fmt::{Display, Formatter};

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
    Exact(Box<Stage38Exact>),
    CertifiedNoSolutionWithinScope(Stage38NoOneExcess),
    Unsupported(Stage38Unsupported),
    ResourceLimit(Stage38ResourceLimit),
}
