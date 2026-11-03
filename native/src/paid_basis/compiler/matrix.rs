use super::super::PaidBasisError;
use crate::state::CnotGate;

pub(super) fn physical_wire(wire: usize) -> Result<u32, PaidBasisError> {
    u32::try_from(wire).map_err(|_| PaidBasisError("wire index exceeds u32 range".into()))
}

pub(super) fn identity_matrix(width: usize) -> Vec<Vec<u8>> {
    (0..width)
        .map(|row| (0..width).map(|column| u8::from(row == column)).collect())
        .collect()
}

pub(super) fn xor_rows(matrix: &mut [Vec<u8>], target: usize, control: usize) {
    let source = matrix[control].clone();
    for (value, source) in matrix[target].iter_mut().zip(source) {
        *value ^= source;
    }
}

fn block_hamming_weight(block: &[Vec<u8>]) -> usize {
    block.iter().flatten().map(|bit| usize::from(*bit)).sum()
}

pub(super) fn paid_block_cost(
    preparation: &[CnotGate],
    block: &[Vec<u8>],
) -> Result<usize, PaidBasisError> {
    preparation
        .len()
        .checked_mul(2)
        .and_then(|prep| prep.checked_add(block_hamming_weight(block)))
        .ok_or_else(|| PaidBasisError("paid partial-block cost overflow".into()))
}
