use crate::i09_quadratic::compiler::slot_width;
use crate::i09_quadratic::{I09Error, I09ErrorKind, I09Monomial};

pub(in crate::i09_quadratic) fn equation_support(
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
