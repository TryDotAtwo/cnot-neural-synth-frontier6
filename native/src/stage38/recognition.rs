use crate::alphabet::CnotAlphabet;
use crate::stage37::{recognize_stage37, Stage37Error, Stage37Result};
use crate::state::MatrixState;

use super::support::{is_complete_unit_alphabet, unsupported};
use super::types::{Stage38Error, Stage38Result};

mod one_excess;
mod one_excess_templates;

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

    one_excess::recognize(state, alphabet, max_entries)
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
