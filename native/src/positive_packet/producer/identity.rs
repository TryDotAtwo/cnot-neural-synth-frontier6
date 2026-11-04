//! Stable producer identity for the pinned stage-59 packet semantics.

use super::super::{OptionalGatePacket, STAGE59_SOURCE_SHA256};
use sha2::{Digest, Sha256};

pub(super) fn producer_version(
    contexts: &[OptionalGatePacket],
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(
        &mut digest,
        b"stage59_optional_packet_v2_source_checked_cache",
    );
    hash_field(&mut digest, STAGE59_SOURCE_SHA256.as_bytes());
    for budget in [max_n, max_slots, max_monomials, max_factor_entries] {
        hash_field(&mut digest, &(budget as u64).to_le_bytes());
    }
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_field(&mut digest, &(context.gates.len() as u64).to_le_bytes());
        for (control, target) in &context.gates {
            hash_field(&mut digest, &control.to_le_bytes());
            hash_field(&mut digest, &target.to_le_bytes());
        }
        hash_field(&mut digest, &(context.factors.len() as u64).to_le_bytes());
        for factor in &context.factors {
            hash_field(&mut digest, &(factor.scope.len() as u64).to_le_bytes());
            for selector in &factor.scope {
                hash_field(&mut digest, &selector.to_le_bytes());
            }
            hash_field(&mut digest, &(factor.allowed.len() as u64).to_le_bytes());
            for pattern in &factor.allowed {
                hash_field(&mut digest, &pattern.to_bytes_le());
            }
        }
    }
    format!(
        "stage59_optional_packet_v2_source_checked_cache:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}
