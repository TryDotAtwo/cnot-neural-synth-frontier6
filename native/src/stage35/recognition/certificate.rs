use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;

use super::super::math::{bit, support_components};
use super::super::types::{Stage35Error, Stage35Exact, Stage35LowerBoundKind, Stage35Result};
use super::super::{STAGE35_PRODUCER, STAGE35_THEOREM_SHA256};
use super::encoding::encode_word;
use super::prepare::PreparedRecognition;

pub(super) fn certify(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    prepared: PreparedRecognition,
) -> Result<Stage35Result, Stage35Error> {
    let PreparedRecognition {
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
    } = prepared;
    let encoder_gates = encode_word(
        branch,
        frame,
        formula_cost,
        active_rows,
        &columns,
        &transformed,
        &shared,
    )?;

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

    Ok(Stage35Result::Exact(Box::new(Stage35Exact {
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
    })))
}
