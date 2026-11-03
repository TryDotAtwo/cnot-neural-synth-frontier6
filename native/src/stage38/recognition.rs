use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::stage37::{recognize_stage37, Stage37Error, Stage37Result};
use crate::state::MatrixState;
use num_bigint::{BigInt, Sign};
use std::collections::HashSet;

use super::contraction::{count_histories, recover_histories};
use super::support::{
    identity_row, is_complete_unit_alphabet, numeric_row_cmp, resource_limit, stage38_lineage,
    unsupported, zero_row,
};
use super::types::{Stage38Error, Stage38Exact, Stage38NoOneExcess, Stage38Result};
use super::xor_rows;
use super::{
    Histories, STAGE38_PRODUCER, STAGE38_SOURCE_IMPLEMENTATION_SHA256, STAGE38_SOURCE_NOTES_SHA256,
};

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

    one_excess(state, alphabet, max_entries)
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

fn one_excess(
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

    let mut total = BigInt::from(0u8);
    let mut templates = 0usize;
    let mut max_bag = 0usize;
    let mut chosen: Option<Histories> = None;

    for target in changed.iter().copied() {
        let mut endpoints = HashSet::new();
        for row in 0..n {
            if row == target {
                continue;
            }
            endpoints.insert(initial[row].clone());
            endpoints.insert(final_rows[row].clone());
        }

        let mut candidates = HashSet::new();
        for endpoint in &endpoints {
            let intermediate = xor_rows(&initial[target], endpoint);
            if intermediate != zero_row(state.words_per_row())
                && intermediate != initial[target]
                && intermediate != final_rows[target]
            {
                candidates.insert(intermediate);
            }
        }
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_by(numeric_row_cmp);

        for intermediate in candidates {
            if !endpoints.contains(&xor_rows(&intermediate, &final_rows[target])) {
                continue;
            }
            let histories: Histories = (0..n)
                .map(|row| {
                    if row == target {
                        vec![
                            initial[row].clone(),
                            intermediate.clone(),
                            final_rows[row].clone(),
                        ]
                    } else if initial[row] != final_rows[row] {
                        vec![initial[row].clone(), final_rows[row].clone()]
                    } else {
                        vec![initial[row].clone()]
                    }
                })
                .collect();
            templates = templates.checked_add(1).ok_or(Stage38Error::SizeOverflow)?;
            let audit = match count_histories(&histories, max_entries, &HashSet::new()) {
                Ok(audit) => audit,
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
            max_bag = max_bag.max(audit.max_bag);
            total += &audit.count;
            if audit.count.sign() == Sign::Plus && chosen.is_none() {
                chosen = Some(histories);
            }
        }
    }

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

    Ok(Stage38Result::Exact(Stage38Exact {
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
    }))
}
