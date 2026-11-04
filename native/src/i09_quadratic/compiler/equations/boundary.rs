//! Final matrix equations tying the last slot to the requested state.

use super::polynomial::Polynomial;
use super::variables::{checked_index, matrix_var};
use crate::i09_quadratic::{I09Context, I09Error, I09Monomial};
use crate::state::MatrixState;

pub(super) fn append_boundary_equations(
    equations: &mut Vec<Vec<I09Monomial>>,
    state: &MatrixState,
    context: &I09Context,
) -> Result<(), I09Error> {
    let n = context.n;
    let last_slot = context.horizon.checked_sub(1);
    for row in 0..n {
        for column in 0..n {
            let mut boundary = Polynomial::new();
            let target_bit = state
                .row_words(row as u32)
                .and_then(|words| words.get(column / 64))
                .is_some_and(|word| ((word >> (column % 64)) & 1) != 0);
            if let Some(last_slot) = last_slot {
                boundary.toggle(I09Monomial::Variable(checked_index(matrix_var(
                    last_slot, row, column, n,
                ))?));
                if target_bit {
                    boundary.toggle(I09Monomial::Constant);
                }
            } else if (row == column) != target_bit {
                boundary.toggle(I09Monomial::Constant);
            }
            equations.push(boundary.finish());
        }
    }
    Ok(())
}
