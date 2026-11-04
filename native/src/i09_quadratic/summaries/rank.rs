use crate::i09_quadratic::{I09Context, I09Error, I09ErrorKind, I09Monomial};
use std::collections::HashMap;

pub(in crate::i09_quadratic) fn rank_beta(
    beta: &[Vec<u8>],
    equation_count: usize,
) -> Result<usize, I09Error> {
    let columns = beta.first().map_or(0, Vec::len);
    let words = equation_count.checked_add(63).ok_or_else(|| {
        I09Error::new(I09ErrorKind::ResourceLimit, "i09_rank_word_count_overflow")
    })? / 64;
    let mut vectors = Vec::new();
    vectors.try_reserve_exact(columns).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_beta_rank_allocation_failed",
        )
    })?;
    for column in 0..columns {
        let mut vector = vec![0u64; words];
        for (equation, row) in beta.iter().enumerate() {
            if row.get(column) == Some(&1) {
                vector[equation / 64] |= 1u64 << (equation % 64);
            }
        }
        vectors.push(vector);
    }
    gf2_rank(vectors, equation_count)
}

pub(in crate::i09_quadratic) fn rank_mixed_terms(
    equations: &[Vec<I09Monomial>],
    context: &I09Context,
    variables: usize,
) -> Result<usize, I09Error> {
    let future_count = variables
        .checked_sub(context.cut)
        .ok_or_else(|| I09Error::new(I09ErrorKind::Internal, "i09_cut_exceeds_variable_order"))?;
    let bit_count = equations.len().checked_mul(future_count).ok_or_else(|| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_mixed_rank_bit_count_overflow",
        )
    })?;
    let words = bit_count.checked_add(63).ok_or_else(|| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_mixed_rank_word_count_overflow",
        )
    })? / 64;
    let mut vectors = Vec::new();
    vectors.try_reserve_exact(context.cut).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_mixed_rank_allocation_failed",
        )
    })?;
    for _ in 0..context.cut {
        vectors.push(vec![0u64; words]);
    }
    for (equation_index, equation) in equations.iter().enumerate() {
        for term in equation {
            let I09Monomial::Product(left, right) = *term else {
                continue;
            };
            let pair = if left < context.cut && right >= context.cut {
                Some((left, right))
            } else if right < context.cut && left >= context.cut {
                Some((right, left))
            } else {
                None
            };
            if let Some((past, future)) = pair {
                let future_index = future - context.cut;
                let bit = equation_index
                    .checked_mul(future_count)
                    .and_then(|base| base.checked_add(future_index))
                    .ok_or_else(|| {
                        I09Error::new(I09ErrorKind::ResourceLimit, "i09_mixed_bit_index_overflow")
                    })?;
                vectors[past][bit / 64] ^= 1u64 << (bit % 64);
            }
        }
    }
    gf2_rank(vectors, bit_count)
}

fn gf2_rank(vectors: Vec<Vec<u64>>, bit_count: usize) -> Result<usize, I09Error> {
    let words = bit_count.checked_add(63).ok_or_else(|| {
        I09Error::new(I09ErrorKind::ResourceLimit, "i09_rank_word_count_overflow")
    })? / 64;
    if vectors.iter().any(|vector| vector.len() != words) {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_rank_vector_has_wrong_width",
        ));
    }
    let mut pivots: HashMap<usize, Vec<u64>> = HashMap::new();
    pivots
        .try_reserve(vectors.len().min(bit_count))
        .map_err(|_| {
            I09Error::new(
                I09ErrorKind::ResourceLimit,
                "i09_rank_pivot_allocation_failed",
            )
        })?;
    for mut vector in vectors {
        for pivot_bit in (0..bit_count).rev() {
            if vector[pivot_bit / 64] & (1u64 << (pivot_bit % 64)) == 0 {
                continue;
            }
            if let Some(pivot) = pivots.get(&pivot_bit) {
                for (word, basis_word) in vector.iter_mut().zip(pivot) {
                    *word ^= basis_word;
                }
            } else {
                pivots.insert(pivot_bit, vector);
                break;
            }
        }
    }
    Ok(pivots.len())
}
