//! Primitive packed-bit operations used by GF(2) matrices.

pub(in crate::bilinear) fn insert_row_basis(pivots: &mut [Option<Vec<u64>>], row: &[u64]) -> bool {
    let mut reduced = row.to_vec();
    for column in (0..pivots.len()).rev() {
        if !get_bit(&reduced, column) {
            continue;
        }
        if let Some(pivot) = &pivots[column] {
            xor_words(&mut reduced, pivot);
        } else {
            pivots[column] = Some(reduced);
            return true;
        }
    }
    false
}

pub(in crate::bilinear) fn words_for(bits: usize) -> Option<usize> {
    bits.checked_add(63).map(|value| value / 64)
}

pub(super) fn get_bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

pub(super) fn set_bit(words: &mut [u64], index: usize) {
    if let Some(word) = words.get_mut(index / 64) {
        *word |= 1u64 << (index % 64);
    }
}

pub(super) fn xor_words(target: &mut [u64], source: &[u64]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target ^= source;
    }
}
