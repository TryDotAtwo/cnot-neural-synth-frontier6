//! Bundle-level validation and conversion to packed matrix states.

use crate::state::MatrixState;

pub(super) fn decode_catalog_state(
    dimension: u32,
    alphabet_id: &str,
    packed_rows: Vec<serde_json::Number>,
) -> Result<MatrixState, String> {
    if packed_rows.len() != dimension as usize {
        return Err("packed row count differs from matrix dimension".into());
    }
    let words_per_row = (dimension as usize)
        .checked_add(63)
        .ok_or_else(|| "packed row width overflow".to_owned())?
        / 64;
    let mut words = Vec::with_capacity(
        (dimension as usize)
            .checked_mul(words_per_row)
            .ok_or_else(|| "packed matrix size overflow".to_owned())?,
    );
    for (row_index, value) in packed_rows.iter().enumerate() {
        let text = value.to_string();
        let bits = num_bigint::BigUint::parse_bytes(text.as_bytes(), 10)
            .ok_or_else(|| format!("row {row_index} is not a nonnegative integer"))?;
        if bits.bits() > u64::from(dimension) {
            return Err(format!("row {row_index} has bits outside matrix width"));
        }
        let row_words = bits.to_u64_digits();
        for word_index in 0..words_per_row {
            words.push(row_words.get(word_index).copied().unwrap_or(0));
        }
    }
    MatrixState::from_packed_rows(dimension, alphabet_id, words).map_err(|error| error.to_string())
}

pub(super) fn catalog_provenance_is_present(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        serde_json::Value::String(value) => !value.trim().is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
    }
}

pub(super) fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
