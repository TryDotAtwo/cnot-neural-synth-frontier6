use super::CheckpointError;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;

pub(super) fn canonical_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_value).collect()),
        Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut result = Map::new();
            for (key, value) in entries {
                result.insert(key, canonical_value(value));
            }
            Value::Object(result)
        }
        other => other,
    }
}

pub(super) fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, CheckpointError> {
    Ok(serde_json::to_vec(value)?)
}

pub(super) fn digest(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    to_hex(&hash)
}

pub(super) fn digest_parts(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    to_hex(&hash.finalize())
}

pub(super) fn digest_reader(reader: &mut File) -> Result<String, CheckpointError> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(to_hex(&hash.finalize()))
}

pub(super) fn read_blob(
    file: &mut File,
    length: u64,
    label: &'static str,
) -> Result<Vec<u8>, CheckpointError> {
    let length = usize::try_from(length)
        .map_err(|_| CheckpointError::Invalid("payload length exceeds address space"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| CheckpointError::AllocationFailed(label))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub(super) fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

pub(super) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
