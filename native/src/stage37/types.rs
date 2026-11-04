use crate::alphabet::{CertificateError, WordCertificate};
use crate::state::MatrixState;
use std::error::Error;
use std::fmt::{Display, Formatter};

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
    Exact(Box<Stage37Exact>),
    NoZeroExcess(Box<Stage37NoZeroExcess>),
    Unsupported(Stage37Unsupported),
}
