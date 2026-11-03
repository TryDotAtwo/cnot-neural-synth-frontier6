//! Fallible workspace allocation and packed-bit primitives.

use super::ComputeFailure;

pub(super) fn words_for(width: usize) -> Option<usize> {
    width.checked_add(63).map(|value| value / 64)
}

pub(super) fn zeroed_u64(length: usize) -> Result<Vec<u64>, ComputeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    values.resize(length, 0);
    Ok(values)
}

pub(super) fn zeroed_f32(length: usize) -> Result<Vec<f32>, ComputeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    values.resize(length, 0.0);
    Ok(values)
}

pub(super) fn set_bit(words: &mut [u64], index: usize) {
    words[index / 64] |= 1u64 << (index % 64);
}

pub(super) fn bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

pub(super) fn highest_set_bit(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().rev().find_map(|(index, word)| {
        (*word != 0).then(|| index * 64 + (63 - word.leading_zeros() as usize))
    })
}

pub(super) fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= right;
    }
}
