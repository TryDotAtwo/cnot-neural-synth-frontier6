use super::types::FrequencyContext;
use super::STAGE52_PRIMARY_SHA256;
use sha2::{Digest, Sha256};

pub(super) fn producer_version(
    contexts: &[FrequencyContext],
    max_width: u32,
    max_frequency_bits: usize,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/stage52-frequency-sector/v1\0");
    hash_bytes(&mut digest, STAGE52_PRIMARY_SHA256.as_bytes());
    hash_u64(&mut digest, max_width as u64);
    hash_u64(&mut digest, max_frequency_bits as u64);
    hash_u64(&mut digest, contexts.len() as u64);
    for context in contexts {
        hash_u64(&mut digest, context.n as u64);
        for block in [&context.sources, &context.internal, &context.targets] {
            hash_u64(&mut digest, block.len() as u64);
            for wire in block {
                hash_u64(&mut digest, *wire as u64);
            }
        }
        hash_u64(&mut digest, context.central_rows.len() as u64);
        for row in &context.central_rows {
            hash_u64(&mut digest, row.len() as u64);
            for word in row {
                hash_u64(&mut digest, *word);
            }
        }
    }
    format!("stage52_frequency_sector_v2:{}", hex(&digest.finalize()))
}

fn hash_bytes(digest: &mut Sha256, value: &[u8]) {
    hash_u64(digest, value.len() as u64);
    digest.update(value);
}

fn hash_u64(digest: &mut Sha256, value: u64) {
    digest.update(value.to_le_bytes());
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}
