use crate::alphabet::{CertificateError, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

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
