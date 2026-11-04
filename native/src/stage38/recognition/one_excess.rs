use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;
use num_bigint::Sign;

use super::super::contraction::recover_histories;
use super::super::support::{identity_row, resource_limit, stage38_lineage};
use super::super::types::{Stage38Error, Stage38Exact, Stage38NoOneExcess, Stage38Result};
use super::super::{
    STAGE38_PRODUCER, STAGE38_SOURCE_IMPLEMENTATION_SHA256, STAGE38_SOURCE_NOTES_SHA256,
};

pub(super) fn recognize(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    max_entries: usize,
) -> Result<Stage38Result, Stage38Error> {
    let n = usize::try_from(state.dimension()).map_err(|_| Stage38Error::SizeOverflow)?;
    let initial = (0..n)
        .map(|row| {
            state
                .row_words(u32::try_from(row).map_err(|_| Stage38Error::SizeOverflow)?)
                .map(<[u64]>::to_vec)
                .ok_or(Stage38Error::SizeOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let final_rows = (0..n)
        .map(|row| identity_row(n, state.words_per_row(), row))
        .collect::<Vec<_>>();
    let changed: Vec<usize> = (0..n)
        .filter(|row| initial[*row] != final_rows[*row])
        .collect();
    let changed_count = changed.len();
    let excluded_length = u64::try_from(changed_count)
        .ok()
        .and_then(|count| count.checked_add(1))
        .ok_or(Stage38Error::SizeOverflow)?;

    let enumeration = match super::one_excess_templates::enumerate(
        state,
        max_entries,
        &initial,
        &final_rows,
        &changed,
    ) {
        Ok(enumeration) => enumeration,
        Err(Stage38Error::ContractionBudgetExceeded(reason)) => {
            return Ok(resource_limit(
                state,
                alphabet,
                max_entries,
                changed_count,
                reason,
            ));
        }
        Err(error) => return Err(error),
    };
    let super::one_excess_templates::Enumeration {
        total,
        templates,
        max_bag,
        chosen,
    } = enumeration;

    if total.sign() == Sign::NoSign {
        return Ok(Stage38Result::CertifiedNoSolutionWithinScope(
            Stage38NoOneExcess {
                state: state.clone(),
                alphabet_id: alphabet.id().to_owned(),
                excluded_length,
                changed_rows: changed_count,
                templates,
                max_bag,
                max_entries,
                source_notes_sha256: STAGE38_SOURCE_NOTES_SHA256,
                source_implementation_sha256: STAGE38_SOURCE_IMPLEMENTATION_SHA256,
            },
        ));
    }

    let chosen = chosen.ok_or(Stage38Error::ContractionInvariant(
        "positive amplitude has no retained history template",
    ))?;
    let mut gates = match recover_histories(&chosen, max_entries) {
        Ok(gates) => gates,
        Err(Stage38Error::ContractionBudgetExceeded(reason)) => {
            return Ok(resource_limit(
                state,
                alphabet,
                max_entries,
                changed_count,
                reason,
            ));
        }
        Err(error) => return Err(error),
    };
    if gates.len()
        != changed_count
            .checked_add(1)
            .ok_or(Stage38Error::SizeOverflow)?
    {
        return Err(Stage38Error::ContractionInvariant(
            "recovered one-excess word has the wrong length",
        ));
    }

    let word = WordCertificate {
        start: state.clone(),
        end: MatrixState::identity(state.dimension(), alphabet.id())
            .map_err(|error| Stage38Error::InvalidCertificate(error.to_string()))?,
        gates: std::mem::take(&mut gates),
        physical_cost: excluded_length,
        producer: STAGE38_PRODUCER.into(),
        lineage_id: stage38_lineage(state)?,
    };
    word.verify(alphabet)
        .map_err(|error| Stage38Error::InvalidCertificate(error.to_string()))?;

    let transport_free =
        (0..n).all(|row| (0..n).all(|other| row == other || initial[row] != final_rows[other]));
    if transport_free && max_bag > 5 {
        return Err(Stage38Error::ContractionInvariant(
            "transport-free width bound was violated",
        ));
    }

    Ok(Stage38Result::Exact(Box::new(Stage38Exact {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        distance: excluded_length,
        changed_rows: changed_count,
        templates,
        one_excess_count: total.to_string(),
        max_bag,
        transport_free,
        max_entries,
        word,
        source_notes_sha256: STAGE38_SOURCE_NOTES_SHA256,
        source_implementation_sha256: STAGE38_SOURCE_IMPLEMENTATION_SHA256,
    })))
}
