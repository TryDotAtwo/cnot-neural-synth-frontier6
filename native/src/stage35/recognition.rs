use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;

use super::types::{Stage35Error, Stage35Result};

mod certificate;
mod encoding;
mod prepare;

/// Recognize the theorem's exact family and return a replayable shortest word.
/// Unsupported inputs have no distance label and are never encoded as zero.
pub fn recognize_stage35(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<Stage35Result, Stage35Error> {
    match prepare::prepare(state, alphabet)? {
        prepare::RecognitionInput::Unsupported(result) => Ok(result),
        prepare::RecognitionInput::Prepared(prepared) => {
            certificate::certify(state, alphabet, prepared)
        }
    }
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
