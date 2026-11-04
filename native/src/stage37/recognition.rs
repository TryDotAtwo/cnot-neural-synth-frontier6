use super::types::{Stage37Error, Stage37Result};
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;

mod certificate;
mod plan;

/// Recognizes words from `state` to identity that attain the changed-row bound.
pub fn recognize_stage37(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<Stage37Result, Stage37Error> {
    match plan::analyze(state, alphabet)? {
        plan::RecognitionPlan::Complete(result) => Ok(result),
        plan::RecognitionPlan::Exact(plan) => certificate::certify(state, alphabet, plan),
    }
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
