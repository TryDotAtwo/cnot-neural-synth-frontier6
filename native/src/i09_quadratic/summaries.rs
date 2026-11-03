//! Exact fixed-prefix alpha/beta summaries and packed GF(2) rank calculations.

use super::compiler::slot_width;
use super::{I09Context, I09Error, I09ErrorKind, I09Monomial};
use std::collections::HashMap;

pub(super) fn compute_alpha(equations: &[Vec<I09Monomial>], context: &I09Context) -> Vec<u8> {
    equations
        .iter()
        .map(|equation| {
            equation.iter().fold(0u8, |value, term| {
                let contribution = match *term {
                    I09Monomial::Constant => Some(1),
                    I09Monomial::Variable(index) if index < context.cut => {
                        Some(context.prefix_bits[index])
                    }
                    I09Monomial::Variable(_) => None,
                    I09Monomial::Product(left, right)
                        if left < context.cut && right < context.cut =>
                    {
                        Some(context.prefix_bits[left] & context.prefix_bits[right])
                    }
                    I09Monomial::Product(_, _) => None,
                };
                value ^ contribution.unwrap_or(0)
            })
        })
        .collect()
}

pub(super) fn compute_beta(
    equations: &[Vec<I09Monomial>],
    context: &I09Context,
    variables: usize,
) -> Result<Vec<Vec<u8>>, I09Error> {
    let future_count = variables
        .checked_sub(context.cut)
        .ok_or_else(|| I09Error::new(I09ErrorKind::Internal, "i09_cut_exceeds_variable_order"))?;
    let mut beta = Vec::new();
    beta.try_reserve_exact(equations.len())
        .map_err(|_| I09Error::new(I09ErrorKind::ResourceLimit, "i09_beta_allocation_failed"))?;
    for equation in equations {
        let mut row = vec![0u8; future_count];
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
                if context.prefix_bits[past] == 1 {
                    let index = future.checked_sub(context.cut).ok_or_else(|| {
                        I09Error::new(I09ErrorKind::Internal, "i09_beta_future_index_underflow")
                    })?;
                    if index >= row.len() {
                        return Err(I09Error::new(
                            I09ErrorKind::Internal,
                            "i09_beta_future_index_out_of_range",
                        ));
                    }
                    row[index] ^= 1;
                }
            }
        }
        beta.push(row);
    }
    Ok(beta)
}

pub(super) fn rank_beta(beta: &[Vec<u8>], equation_count: usize) -> Result<usize, I09Error> {
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

pub(super) fn rank_mixed_terms(
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

pub(super) fn equation_support(
    equations: &[Vec<I09Monomial>],
    n: usize,
    horizon: usize,
    variables: usize,
) -> Result<Vec<u8>, I09Error> {
    let support_len = equations
        .len()
        .checked_mul(n)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_support_size_overflow"))?;
    let mut support = vec![0u8; support_len];
    for (equation_index, equation) in equations.iter().enumerate() {
        for term in equation {
            match *term {
                I09Monomial::Constant => {}
                I09Monomial::Variable(index) => add_variable_support(
                    &mut support,
                    equation_index,
                    index,
                    n,
                    horizon,
                    variables,
                )?,
                I09Monomial::Product(left, right) => {
                    add_variable_support(
                        &mut support,
                        equation_index,
                        left,
                        n,
                        horizon,
                        variables,
                    )?;
                    add_variable_support(
                        &mut support,
                        equation_index,
                        right,
                        n,
                        horizon,
                        variables,
                    )?;
                }
            }
        }
    }
    Ok(support)
}

fn add_variable_support(
    support: &mut [u8],
    equation: usize,
    index: usize,
    n: usize,
    horizon: usize,
    variables: usize,
) -> Result<(), I09Error> {
    if index >= variables || index / slot_width(n).unwrap_or(0) >= horizon {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_support_variable_index_out_of_range",
        ));
    }
    let slot_size = slot_width(n)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_slot_width_overflow"))?;
    let within = index % slot_size;
    let base = equation
        .checked_mul(n)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_support_index_overflow"))?;
    if within == 0 {
        support[base..base + n].fill(1);
    } else if within <= n {
        support[base + within - 1] = 1;
    } else if within <= 2 * n {
        support[base + within - n - 1] = 1;
    } else if within <= 3 * n {
        support[base + within - 2 * n - 1] = 1;
    } else {
        let offset = within - (1 + 3 * n);
        let row = offset / n;
        let column = offset % n;
        support[base + row] = 1;
        support[base + column] = 1;
    }
    Ok(())
}

pub(super) fn variable_names(
    n: usize,
    horizon: usize,
    variables: usize,
) -> Result<Vec<String>, I09Error> {
    let mut names = Vec::new();
    names.try_reserve_exact(variables).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_variable_names_allocation_failed",
        )
    })?;
    for slot in 0..horizon {
        names.push(format!("active[{slot}]"));
        for wire in 0..n {
            names.push(format!("control[{slot},{wire}]"));
        }
        for wire in 0..n {
            names.push(format!("target[{slot},{wire}]"));
        }
        for wire in 0..n {
            names.push(format!("readout[{slot},{wire}]"));
        }
        for row in 0..n {
            for column in 0..n {
                names.push(format!("matrix[{slot},{row},{column}]"));
            }
        }
    }
    if names.len() != variables {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_variable_name_count_mismatch",
        ));
    }
    Ok(names)
}
