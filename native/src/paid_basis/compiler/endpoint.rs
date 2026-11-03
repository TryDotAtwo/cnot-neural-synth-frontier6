use super::super::PaidBasisError;
use crate::state::MatrixState;
use sha2::{Digest, Sha256};

pub(crate) fn endpoint_sha256(state: &MatrixState) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"cnot103_paid_shear_endpoint_v1");
    hash_field(&mut digest, &state.dimension().to_le_bytes());
    hash_field(&mut digest, state.alphabet_id().as_bytes());
    hash_field(
        &mut digest,
        &(state.packed_rows().len() as u64).to_le_bytes(),
    );
    for word in state.packed_rows() {
        hash_field(&mut digest, &word.to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_field(digest: &mut Sha256, field: &[u8]) {
    digest.update((field.len() as u64).to_le_bytes());
    digest.update(field);
}

pub(super) fn expected_shear_endpoint(
    start: &MatrixState,
    dimension: u32,
    source_wires: &[usize],
    target_wires: &[usize],
    block: &[Vec<u8>],
) -> Result<MatrixState, PaidBasisError> {
    let words_per_row = start.words_per_row();
    let row_count = usize::try_from(dimension)
        .map_err(|_| PaidBasisError("dimension cannot be represented as usize".into()))?;
    let packed_len = row_count
        .checked_mul(words_per_row)
        .ok_or_else(|| PaidBasisError("expected endpoint allocation overflow".into()))?;
    let mut rows = vec![0u64; packed_len];
    for wire in 0..row_count {
        rows[wire * words_per_row + wire / 64] |= 1u64 << (wire % 64);
    }
    for (target_index, target_wire) in target_wires.iter().copied().enumerate() {
        for (source_index, source_wire) in source_wires.iter().copied().enumerate() {
            if block[target_index][source_index] != 0 {
                let offset = target_wire * words_per_row + source_wire / 64;
                rows[offset] ^= 1u64 << (source_wire % 64);
            }
        }
    }
    MatrixState::from_packed_rows(dimension, start.alphabet_id(), rows)
        .map_err(|error| PaidBasisError(error.to_string()))
}
