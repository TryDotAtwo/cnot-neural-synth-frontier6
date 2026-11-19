//! Matrix-only lower bounds for emitted exact roots. The caller has already
//! verified a word of the claimed cost. Source strings and witnesses in a
//! training record never establish a lower bound.
use super::TrainingDataError;
use crate::stage35::{bit, is_full_unit_alphabet, is_identity_row, support_components};
use crate::{
    recognize_stage35, recognize_stage37, CnotAlphabet, MatrixState, Stage35Result, Stage37Result,
};

mod graph;

pub(super) fn proves(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    distance: u64,
) -> Result<bool, TrainingDataError> {
    if !is_full_unit_alphabet(alphabet).map_err(|e| TrainingDataError(e.to_string()))? {
        return Ok(false);
    }
    let n = state.dimension() as usize;
    let changed: Vec<_> = (0..n).filter(|&i| !is_identity_row(state, i)).collect();
    if changed.len() <= 2 && n >= 2 {
        let result =
            recognize_stage35(state, alphabet).map_err(|e| TrainingDataError(e.to_string()))?;
        return Ok(matches!(result, Stage35Result::Exact(exact) if exact.distance == distance));
    }
    if distance == changed.len() as u64 + 1 {
        // The checked word supplies r+1. Excluding r now proves optimality
        // without constructing or counting the one-excess words.
        let result =
            recognize_stage37(state, alphabet).map_err(|e| TrainingDataError(e.to_string()))?;
        if matches!(result, Stage37Result::NoZeroExcess(no) if no.excluded_length + 1 == distance) {
            return Ok(true);
        }
    }
    if changed.len() == 3 && is_stage36_frame(state, &changed) {
        let diagonal_zeros = changed.iter().filter(|&&row| !bit(state, row, row)).count();
        let lower = n - support_components(state) + diagonal_zeros;
        if lower as u64 == distance {
            return Ok(true);
        }
    }
    if changed.len() + 1 != n {
        return Ok(false);
    }
    Ok(graph::lower_bound(state) == Some(distance))
}

fn is_stage36_frame(state: &MatrixState, changed: &[usize]) -> bool {
    // The B* internal frame (6, 5, 7), up to a joint relabeling of its
    // three physical wires, is the scope of the stage36 source theorem.
    [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    .into_iter()
    .any(|p| {
        let wires = p.map(|i| changed[i]);
        let frame = wires.map(|i| {
            wires.iter().enumerate().fold(0u8, |mask, (j, &k)| {
                mask | (u8::from(bit(state, i, k)) << j)
            })
        });
        frame == [6, 5, 7]
    })
}
