use super::math::{
    bit, frame_for, gate, inverse_2x2, is_full_unit_alphabet, is_identity_row, multiply_2x2,
    restricted_row, support_components, FRAME_IDENTITY, FRAME_IDENTITY_WITH_SHARED, IDENTITY_FRAME,
};
use super::types::{
    Stage35Branch, Stage35Error, Stage35Exact, Stage35LowerBoundKind, Stage35Result,
    Stage35Unsupported,
};
use super::{STAGE35_PRODUCER, STAGE35_THEOREM_SHA256};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;

/// Recognize the theorem's exact family and return a replayable shortest word.
/// Unsupported inputs have no distance label and are never encoded as zero.
pub fn recognize_stage35(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<Stage35Result, Stage35Error> {
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

    let mut encoder_gates = Vec::with_capacity(formula_cost);
    match branch {
        Stage35Branch::NoFrameMotion => {
            for (column, value) in columns {
                if value & 1 != 0 {
                    encoder_gates.push(gate(column, active[0])?);
                }
                if value & 2 != 0 {
                    encoder_gates.push(gate(column, active[1])?);
                }
            }
        }
        Stage35Branch::NonemptyFrameMotion => {
            for (column, value) in &transformed {
                if *value == 1 || *value == 2 {
                    let target = if *value == 1 { active[0] } else { active[1] };
                    encoder_gates.push(gate(*column, target)?);
                }
            }
            let (control, target) = frame[0];
            encoder_gates.push(gate(active[control], active[target])?);

            let mut first_frame = IDENTITY_FRAME;
            first_frame[target] ^= first_frame[control];
            let first_inverse = inverse_2x2(first_frame)?;
            let receiver = (0..2)
                .find(|row| multiply_2x2(first_inverse, 1u8 << row) == 3)
                .ok_or(Stage35Error::MissingFrame(first_frame))?;
            for column in &shared {
                encoder_gates.push(gate(*column, active[receiver])?);
            }
            for (control, target) in frame.iter().copied().skip(1) {
                encoder_gates.push(gate(active[control], active[target])?);
            }
        }
    }

    if encoder_gates.len() != formula_cost {
        return Err(Stage35Error::FormulaWordLengthMismatch {
            formula: u64::try_from(formula_cost).map_err(|_| Stage35Error::CostOverflow)?,
            word: encoder_gates.len(),
        });
    }
    let distance = u64::try_from(formula_cost).map_err(|_| Stage35Error::CostOverflow)?;
    let identity = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|_| Stage35Error::AlphabetMismatch)?;
    let word = WordCertificate {
        start: state.clone(),
        end: identity,
        gates: encoder_gates.iter().rev().copied().collect(),
        physical_cost: distance,
        producer: STAGE35_PRODUCER.into(),
        lineage_id: format!("stage35:{}:{:?}", state.dimension(), state.packed_rows()),
    };
    word.verify(alphabet)?;

    let connectivity = state.dimension() as usize - support_components(state);
    let zero_diagonal = (0..n).filter(|index| !bit(state, *index, *index)).count();
    let (lower_bound, lower_bound_kind) = if !identity_frame {
        (
            connectivity
                .checked_add(zero_diagonal)
                .ok_or(Stage35Error::CostOverflow)?,
            Stage35LowerBoundKind::ConnectivityPlusDiagonalZeros,
        )
    } else if shared.len() >= 2 {
        (
            connectivity
                .checked_add(1)
                .ok_or(Stage35Error::CostOverflow)?,
            Stage35LowerBoundKind::TreeObstructionTwoSharedSources,
        )
    } else {
        (connectivity, Stage35LowerBoundKind::Connectivity)
    };
    let lower_bound = u64::try_from(lower_bound).map_err(|_| Stage35Error::CostOverflow)?;
    if lower_bound != distance {
        return Err(Stage35Error::TheoremMismatch {
            distance,
            lower_bound,
        });
    }

    let changed_rows = changed_rows
        .into_iter()
        .map(|row| u32::try_from(row).map_err(|_| Stage35Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let active_rows = [
        u32::try_from(active_rows[0]).map_err(|_| Stage35Error::IndexOverflow)?,
        u32::try_from(active_rows[1]).map_err(|_| Stage35Error::IndexOverflow)?,
    ];
    let shared_sources = shared
        .into_iter()
        .take(2)
        .map(|row| u32::try_from(row).map_err(|_| Stage35Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Stage35Result::Exact(Stage35Exact {
        alphabet_id: alphabet.id().to_owned(),
        distance,
        word,
        theorem_sha256: STAGE35_THEOREM_SHA256,
        changed_rows,
        active_rows,
        frame_rows,
        nonzero_external_columns,
        shared_external_columns,
        branch,
        lower_bound_kind,
        connectivity,
        zero_diagonal,
        shared_sources,
        lower_bound,
        encoder_gates,
    }))
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

fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> Stage35Result {
    Stage35Result::Unsupported(Stage35Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    })
}
