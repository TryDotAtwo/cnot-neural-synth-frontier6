//! Memory and factor-entry admission estimates for the I08 producer.

use super::polynomial::PolynomialTerm;
use super::storage::words_for;
use crate::positive_packet::LocalBooleanFactor;
use std::mem::size_of;

pub(in crate::positive_packet) fn working_bytes_upper_bound(
    n: usize,
    slots: usize,
    max_monomials: usize,
) -> Option<usize> {
    let words_per_row = n.checked_add(63)?.checked_div(64)?;
    let matrix_bytes = n
        .checked_mul(words_per_row)?
        .checked_mul(size_of::<u64>())?;
    let mask_bytes = words_for(slots)?.checked_mul(size_of::<u64>())?;
    let per_term = matrix_bytes
        .checked_add(mask_bytes)?
        .checked_add(size_of::<PolynomialTerm>())?
        .checked_add(128)?;
    let polynomial = max_monomials.checked_mul(per_term)?.checked_mul(2)?;
    let flat_words = words_for(n.checked_mul(n)?)?;
    let rank_workspace = slots
        .checked_mul(flat_words)?
        .checked_mul(size_of::<u64>())?
        .checked_mul(2)?;
    let groups = slots.checked_add(1)?;
    let output = groups
        .checked_mul(
            4usize
                .checked_add(n.checked_mul(2)?)?
                .checked_mul(size_of::<f32>())?,
        )?
        .checked_add(groups)?;
    polynomial.checked_add(rank_workspace)?.checked_add(output)
}

pub(in crate::positive_packet) fn factor_entry_count(
    factors: &[LocalBooleanFactor],
) -> Option<usize> {
    factors.iter().try_fold(0usize, |total, factor| {
        total
            .checked_add(factor.scope.len())?
            .checked_add(factor.allowed.len())
    })
}
