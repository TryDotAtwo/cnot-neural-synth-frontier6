use crate::i09_quadratic::I09Monomial;
use std::mem::size_of;

pub(in crate::i09_quadratic) fn variable_count(n: usize, horizon: usize) -> Option<usize> {
    let per_slot = n
        .checked_mul(n)?
        .checked_add(n.checked_mul(3)?)?
        .checked_add(1)?;
    horizon.checked_mul(per_slot)
}

pub(in crate::i09_quadratic) fn equation_bound(n: usize, horizon: usize) -> Option<usize> {
    let n_squared = n.checked_mul(n)?;
    let per_slot = 3usize
        .checked_add(n.checked_mul(n.checked_sub(1)?)?)?
        .checked_add(n_squared.checked_mul(2)?)?
        .checked_add(n)?;
    horizon.checked_mul(per_slot)?.checked_add(n_squared)
}

pub(in crate::i09_quadratic) fn working_bytes_upper_bound(
    n: usize,
    horizon: usize,
    variables: usize,
    equations: usize,
) -> Option<usize> {
    let channels = variables.checked_add(4)?;
    let feature_bytes = equations
        .checked_mul(channels.checked_add(n)?)?
        .checked_mul(size_of::<f32>())?;
    let terms = equations.checked_mul(n.checked_add(2)?)?;
    let equation_bytes = terms
        .checked_mul(size_of::<I09Monomial>())?
        .checked_add(equations.checked_mul(size_of::<Vec<I09Monomial>>())?)?;
    let mixed_bits = equations.checked_mul(variables)?;
    let mixed_words = mixed_bits.checked_add(63)?.checked_div(64)?;
    let rank_bytes = variables
        .checked_mul(mixed_words)?
        .checked_mul(size_of::<u64>())?
        .checked_add(
            variables
                .checked_mul(equations.checked_add(63)?.checked_div(64)?)?
                .checked_mul(size_of::<u64>())?,
        )?;
    let metadata_bytes = variables
        .checked_mul(64)?
        .checked_add(equations.checked_mul(size_of::<u8>())?)?
        .checked_add(n.checked_mul(horizon)?)?;
    feature_bytes
        .checked_add(equation_bytes)?
        .checked_add(rank_bytes)?
        .checked_add(metadata_bytes)
}

pub(in crate::i09_quadratic) fn slot_width(n: usize) -> Option<usize> {
    n.checked_mul(n)?
        .checked_add(n.checked_mul(3)?)?
        .checked_add(1)
}
