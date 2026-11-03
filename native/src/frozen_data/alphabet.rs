//! Canonical CNOT alphabet construction shared by wire decoding and symmetry checks.
use super::types::FrozenDataError;
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;

pub(super) fn alphabet(s: &MatrixState) -> Result<CnotAlphabet, FrozenDataError> {
    CnotAlphabet::all_to_all_unit(s.dimension()).map_err(|e| FrozenDataError(e.to_string()))
}
