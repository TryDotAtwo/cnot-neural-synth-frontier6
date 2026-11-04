//! Conservative pre-allocation estimate for I06 relation tensors.

pub(super) fn working_bytes_upper_bound(
    n: usize,
    vectors: usize,
    coordinates: usize,
    ports: usize,
    max_vectors: usize,
) -> Option<usize> {
    let square = n.checked_mul(n)?;
    let context_work = coordinates
        .checked_mul(vectors)?
        .checked_add(vectors.checked_mul(vectors)?)?
        .checked_mul(16)?;
    let group_work = vectors
        .checked_add(ports)?
        .checked_mul(max_vectors.checked_add(2)?.checked_add(n.checked_mul(2)?)?)?
        .checked_mul(4)?;
    square
        .checked_mul(12)?
        .checked_add(context_work)?
        .checked_add(group_work)
}
