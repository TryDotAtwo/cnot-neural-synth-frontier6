//! Verified all-to-all unit-cost CNOT upper-bound compiler.
//!
//! The row-reduction order is deterministic but not shortest. The returned
//! word is replayed against the full matrix; its length is an upper bound only.

use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;
use std::error::Error;
use std::fmt::{Display, Formatter};

mod row_reduction;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneralCompilerLimits {
    pub max_dimension: u32,
    pub max_gates: u64,
}

impl Default for GeneralCompilerLimits {
    fn default() -> Self {
        Self {
            max_dimension: 1024,
            max_gates: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GeneralCompilerError {
    Unsupported(String),
    ResourceLimit(String),
    Invariant(String),
}

impl Display for GeneralCompilerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(formatter, "unsupported: {reason}"),
            Self::ResourceLimit(reason) => write!(formatter, "resource limit: {reason}"),
            Self::Invariant(reason) => write!(formatter, "compiler invariant: {reason}"),
        }
    }
}

impl Error for GeneralCompilerError {}

/// Compiles an invertible matrix to a replayable word over the all-to-all,
/// unit-cost CNOT alphabet. Row swaps use three CNOTs. The deterministic
/// Gauss-Jordan word is a verified upper bound and carries no optimality claim.
pub fn compile_general_cnot_word(
    state: &MatrixState,
    limits: GeneralCompilerLimits,
) -> Result<WordCertificate, GeneralCompilerError> {
    let n = state.dimension();
    if limits.max_dimension == 0 || n > limits.max_dimension {
        return Err(GeneralCompilerError::ResourceLimit(format!(
            "matrix width {n} exceeds compiler limit {}",
            limits.max_dimension
        )));
    }

    let alphabet = CnotAlphabet::all_to_all_unit(n)
        .map_err(|error| GeneralCompilerError::Unsupported(error.to_string()))?;
    if state.alphabet_id() != alphabet.id() {
        return Err(GeneralCompilerError::Unsupported(
            "compiler requires the all-to-all unit-cost CNOT alphabet".into(),
        ));
    }

    let n_usize = usize::try_from(n).map_err(|_| {
        GeneralCompilerError::ResourceLimit("width exceeds host index range".into())
    })?;
    let gates = row_reduction::to_identity(state, n_usize, limits.max_gates)?;
    let gates = gates.into_iter().rev().collect::<Vec<_>>();
    let physical_cost = u64::try_from(gates.len())
        .map_err(|_| GeneralCompilerError::ResourceLimit("word length exceeds u64".into()))?;
    let certificate = WordCertificate {
        start: MatrixState::identity(n, alphabet.id())
            .map_err(|error| GeneralCompilerError::Invariant(error.to_string()))?,
        end: state.clone(),
        gates,
        physical_cost,
        producer: "general_gauss_jordan_v1".into(),
        lineage_id: "full_matrix_row_reduction".into(),
    };
    let replayed = certificate
        .verify(&alphabet)
        .map_err(|error| GeneralCompilerError::Invariant(error.to_string()))?;
    if &replayed != state {
        return Err(GeneralCompilerError::Invariant(
            "replayed CNOT word differs from the requested matrix".into(),
        ));
    }
    Ok(certificate)
}
