use crate::i09_quadratic::{I09Context, I09Error, I09ErrorKind, I09Monomial};

pub(in crate::i09_quadratic) fn compute_alpha(
    equations: &[Vec<I09Monomial>],
    context: &I09Context,
) -> Vec<u8> {
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

pub(in crate::i09_quadratic) fn compute_beta(
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
