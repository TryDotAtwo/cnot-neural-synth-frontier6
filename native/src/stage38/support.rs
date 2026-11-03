use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use num_bigint::BigInt;

use super::types::{Stage38Error, Stage38ResourceLimit, Stage38Result, Stage38Unsupported};
use super::Row;

pub(super) fn is_complete_unit_alphabet(alphabet: &CnotAlphabet) -> bool {
    let Ok(expected) = CnotAlphabet::all_to_all_unit(alphabet.dimension()) else {
        return false;
    };
    alphabet.id() == expected.id()
        && alphabet.gates() == expected.gates()
        && alphabet
            .gates()
            .iter()
            .enumerate()
            .all(|(index, _)| alphabet.cost_at(index) == Some(1))
}

pub(super) fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> Stage38Result {
    Stage38Result::Unsupported(Stage38Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    })
}

pub(super) fn resource_limit(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    max_entries: usize,
    changed_rows: usize,
    reason: String,
) -> Stage38Result {
    Stage38Result::ResourceLimit(Stage38ResourceLimit {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        max_entries,
        changed_rows,
        reason,
    })
}

pub(super) fn xor_rows(left: &Row, right: &Row) -> Row {
    left.iter()
        .zip(right)
        .map(|(left, right)| left ^ right)
        .collect()
}

pub(super) fn zero_row(words: usize) -> Row {
    vec![0; words]
}

pub(super) fn identity_row(n: usize, words: usize, row: usize) -> Row {
    let mut value = vec![0; words];
    value[row / 64] = 1u64 << (row % 64);
    debug_assert!(row < n);
    value
}

pub(super) fn numeric_row_cmp(left: &Row, right: &Row) -> std::cmp::Ordering {
    left.iter().rev().cmp(right.iter().rev())
}

pub(super) fn stage38_lineage(state: &MatrixState) -> Result<String, Stage38Error> {
    let n = usize::try_from(state.dimension()).map_err(|_| Stage38Error::SizeOverflow)?;
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let words = state
            .row_words(u32::try_from(row).map_err(|_| Stage38Error::SizeOverflow)?)
            .ok_or(Stage38Error::SizeOverflow)?;
        let mut value = BigInt::from(0u8);
        for word in words.iter().rev() {
            value <<= 64usize;
            value += BigInt::from(*word);
        }
        rows.push(value.to_string());
    }
    let tuple = if rows.len() == 1 {
        format!("({},)", rows[0])
    } else {
        format!("({})", rows.join(", "))
    };
    Ok(format!("stage38:{}:{tuple}", state.dimension()))
}
