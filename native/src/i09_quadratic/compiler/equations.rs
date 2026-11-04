//! Ordered equation assembly for the I09 quadratic synthesizer.

mod boundary;
mod polynomial;
mod slot;
mod variables;

use super::layout::equation_bound;
use crate::i09_quadratic::{I09Context, I09Error, I09ErrorKind, I09Monomial};
use crate::state::MatrixState;

pub(in crate::i09_quadratic) fn compile_equations(
    state: &MatrixState,
    context: &I09Context,
    variable_count: usize,
) -> Result<Vec<Vec<I09Monomial>>, I09Error> {
    let n = context.n;
    let mut equations = Vec::new();
    equations
        .try_reserve_exact(equation_bound(n, context.horizon).unwrap_or_default())
        .map_err(|_| {
            I09Error::new(
                I09ErrorKind::ResourceLimit,
                "i09_equation_allocation_failed",
            )
        })?;
    for slot in 0..context.horizon {
        slot::append_slot_equations(&mut equations, slot, context)?;
    }
    boundary::append_boundary_equations(&mut equations, state, context)?;
    if equations
        .iter()
        .flat_map(|equation| equation.iter())
        .any(|term| !polynomial::monomial_is_in_range(*term, variable_count))
    {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_compiler_emitted_an_out_of_range_variable",
        ));
    }
    Ok(equations)
}
