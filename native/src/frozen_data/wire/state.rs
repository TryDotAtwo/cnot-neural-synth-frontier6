//! Decoding of packed matrix states and forbidden-key sets.

use super::super::types::FrozenDataError;
use super::model::{Key, State};
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::HashSet;

const MAX_N: u32 = 256;

pub(in crate::frozen_data) fn decode_state(raw: State) -> Result<MatrixState, FrozenDataError> {
    if raw.n == 0 || raw.n > MAX_N {
        return Err(FrozenDataError(format!(
            "unsupported matrix dimension {}",
            raw.n
        )));
    }
    let alpha = CnotAlphabet::all_to_all_unit(raw.n).map_err(|e| FrozenDataError(e.to_string()))?;
    if raw.alphabet_id != alpha.id() || raw.packed_rows.len() != raw.n as usize {
        return Err(FrozenDataError(
            "matrix alphabet or row count mismatch".into(),
        ));
    }
    let width = (raw.n as usize + 63) / 64;
    let mut packed = Vec::with_capacity(raw.n as usize * width);
    for (i, value) in raw.packed_rows.iter().enumerate() {
        let text = value.to_string();
        let bits = BigUint::parse_bytes(text.as_bytes(), 10)
            .ok_or_else(|| FrozenDataError(format!("row {i} is not a nonnegative integer")))?;
        if bits.bits() > u64::from(raw.n) {
            return Err(FrozenDataError(format!(
                "row {i} has bits outside matrix width"
            )));
        }
        let words = bits.to_u64_digits();
        for word in 0..width {
            packed.push(words.get(word).copied().unwrap_or(0));
        }
    }
    MatrixState::from_packed_rows(raw.n, raw.alphabet_id, packed)
        .map_err(|e| FrozenDataError(e.to_string()))
}

pub(in crate::frozen_data) fn key_set(
    keys: Vec<Key>,
    name: &str,
) -> Result<HashSet<MatrixState>, FrozenDataError> {
    let mut out = HashSet::with_capacity(keys.len());
    for (n, alphabet_id, packed_rows) in keys {
        let state = decode_state(State {
            n,
            alphabet_id,
            packed_rows,
        })
        .map_err(|e| FrozenDataError(format!("{name}: {e}")))?;
        if !out.insert(state) {
            return Err(FrozenDataError(format!("{name} has duplicate keys")));
        }
    }
    Ok(out)
}
