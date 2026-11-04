use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;

use super::super::math::{
    bit, frame_for, inverse_2x2, is_full_unit_alphabet, is_identity_row, multiply_2x2,
    restricted_row, FRAME_IDENTITY, FRAME_IDENTITY_WITH_SHARED, IDENTITY_FRAME,
};
use super::super::types::{Stage35Branch, Stage35Error, Stage35Result, Stage35Unsupported};

pub(super) enum RecognitionInput {
    Unsupported(Stage35Result),
    Prepared(PreparedRecognition),
}

pub(super) struct PreparedRecognition {
    pub(super) n: usize,
    pub(super) identity_frame: bool,
    pub(super) changed_rows: Vec<usize>,
    pub(super) active_rows: [usize; 2],
    pub(super) frame_rows: [u8; 2],
    pub(super) columns: Vec<(usize, u8)>,
    pub(super) transformed: Vec<(usize, u8)>,
    pub(super) shared: Vec<usize>,
    pub(super) nonzero_external_columns: usize,
    pub(super) shared_external_columns: usize,
    pub(super) branch: Stage35Branch,
    pub(super) frame: &'static [(usize, usize)],
    pub(super) formula_cost: usize,
}

pub(super) fn prepare(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<RecognitionInput, Stage35Error> {
    if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
        return Err(Stage35Error::AlphabetMismatch);
    }
    if !is_full_unit_alphabet(alphabet)? {
        return Ok(unsupported(
            state,
            alphabet,
            "requires full unit CNOT alphabet",
        ));
    }
    let n = usize::try_from(state.dimension()).map_err(|_| Stage35Error::IndexOverflow)?;
    if n < 2 {
        return Ok(unsupported(state, alphabet, "requires at least two wires"));
    }

    let changed_rows: Vec<usize> = (0..n).filter(|row| !is_identity_row(state, *row)).collect();
    if changed_rows.len() > 2 {
        return Ok(unsupported(state, alphabet, "more than two changed rows"));
    }

    let mut active = changed_rows.clone();
    for row in 0..n {
        if active.len() == 2 {
            break;
        }
        if !active.contains(&row) {
            active.push(row);
        }
    }
    let active_rows = [active[0], active[1]];
    let frame_rows = [
        restricted_row(state, active[0], active_rows),
        restricted_row(state, active[1], active_rows),
    ];
    let frame_inverse = inverse_2x2(frame_rows)?;
    let external: Vec<usize> = (0..n).filter(|row| !active.contains(row)).collect();

    let columns: Vec<(usize, u8)> = external
        .iter()
        .copied()
        .map(|column| {
            let value = u8::from(bit(state, active[0], column))
                | (u8::from(bit(state, active[1], column)) << 1);
            (column, value)
        })
        .collect();
    let transformed: Vec<(usize, u8)> = columns
        .iter()
        .map(|(column, value)| (*column, multiply_2x2(frame_inverse, *value)))
        .collect();
    let nonzero_external_columns = transformed.iter().filter(|(_, value)| *value != 0).count();
    let shared: Vec<usize> = transformed
        .iter()
        .filter_map(|(column, value)| (*value == 3).then_some(*column))
        .collect();
    let shared_external_columns = shared.len();
    let identity_frame = frame_rows == IDENTITY_FRAME;

    let (branch, frame, formula_cost) = if identity_frame && shared.len() <= 2 {
        let cost = nonzero_external_columns
            .checked_add(shared.len())
            .ok_or(Stage35Error::CostOverflow)?;
        (Stage35Branch::NoFrameMotion, FRAME_IDENTITY, cost)
    } else {
        let frame = if identity_frame {
            FRAME_IDENTITY_WITH_SHARED
        } else {
            frame_for(frame_rows).ok_or(Stage35Error::MissingFrame(frame_rows))?
        };
        let cost = nonzero_external_columns
            .checked_add(frame.len())
            .ok_or(Stage35Error::CostOverflow)?;
        (Stage35Branch::NonemptyFrameMotion, frame, cost)
    };

    Ok(RecognitionInput::Prepared(PreparedRecognition {
        n,
        identity_frame,
        changed_rows,
        active_rows,
        frame_rows,
        columns,
        transformed,
        shared,
        nonzero_external_columns,
        shared_external_columns,
        branch,
        frame,
        formula_cost,
    }))
}

fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> RecognitionInput {
    RecognitionInput::Unsupported(Stage35Result::Unsupported(Stage35Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    }))
}
