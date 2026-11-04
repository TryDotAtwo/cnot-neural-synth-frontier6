use super::layout::{equation_bound, slot_width};
use crate::i09_quadratic::{I09Context, I09Error, I09ErrorKind, I09Monomial};
use crate::state::MatrixState;
use std::collections::BTreeSet;

fn active_var(slot: usize, n: usize) -> Option<usize> {
    slot.checked_mul(slot_width(n)?)
}

fn control_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    active_var(slot, n)?.checked_add(1)?.checked_add(wire)
}

fn target_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    control_var(slot, n, n)?.checked_add(wire)
}

fn readout_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    control_var(slot, n.checked_mul(2)?, n)?.checked_add(wire)
}

fn matrix_var(slot: usize, row: usize, column: usize, n: usize) -> Option<usize> {
    let offset = row.checked_mul(n)?.checked_add(column)?;
    readout_var(slot, n, n)?.checked_add(offset)
}

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
        let active = checked_index(active_var(slot, n))?;
        let mut active_controls = Polynomial::new();
        active_controls.toggle(I09Monomial::Variable(active));
        let mut active_targets = Polynomial::new();
        active_targets.toggle(I09Monomial::Variable(active));
        for wire in 0..n {
            active_controls.toggle(I09Monomial::Variable(checked_index(control_var(
                slot, wire, n,
            ))?));
            active_targets.toggle(I09Monomial::Variable(checked_index(target_var(
                slot, wire, n,
            ))?));
        }
        equations.push(active_controls.finish());
        equations.push(active_targets.finish());

        for left in 0..n {
            for right in left + 1..n {
                equations.push(vec![product(
                    checked_index(control_var(slot, left, n))?,
                    checked_index(control_var(slot, right, n))?,
                )]);
            }
        }
        for left in 0..n {
            for right in left + 1..n {
                equations.push(vec![product(
                    checked_index(target_var(slot, left, n))?,
                    checked_index(target_var(slot, right, n))?,
                )]);
            }
        }
        for control in 0..n {
            for target in 0..n {
                if !context.allowed_gates.contains(&(control, target)) {
                    equations.push(vec![product(
                        checked_index(control_var(slot, control, n))?,
                        checked_index(target_var(slot, target, n))?,
                    )]);
                }
            }
        }
        if slot > 0 {
            let mut monotone = Polynomial::new();
            let current = checked_index(active_var(slot, n))?;
            let previous = checked_index(active_var(slot - 1, n))?;
            monotone.toggle(I09Monomial::Variable(current));
            monotone.toggle(product(previous, current));
            equations.push(monotone.finish());
        }
        for column in 0..n {
            let mut readout = Polynomial::new();
            readout.toggle(I09Monomial::Variable(checked_index(readout_var(
                slot, column, n,
            ))?));
            for row in 0..n {
                let control = checked_index(control_var(slot, row, n))?;
                if slot == 0 {
                    if row == column {
                        readout.toggle(I09Monomial::Variable(control));
                    }
                } else {
                    readout.toggle(product(
                        control,
                        checked_index(matrix_var(slot - 1, row, column, n))?,
                    ));
                }
            }
            equations.push(readout.finish());
        }
        for row in 0..n {
            for column in 0..n {
                let mut update = Polynomial::new();
                update.toggle(I09Monomial::Variable(checked_index(matrix_var(
                    slot, row, column, n,
                ))?));
                if slot == 0 {
                    if row == column {
                        update.toggle(I09Monomial::Constant);
                    }
                } else {
                    update.toggle(I09Monomial::Variable(checked_index(matrix_var(
                        slot - 1,
                        row,
                        column,
                        n,
                    ))?));
                }
                update.toggle(product(
                    checked_index(target_var(slot, row, n))?,
                    checked_index(readout_var(slot, column, n))?,
                ));
                equations.push(update.finish());
            }
        }
    }
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
                // With no CNOT slots, the compiler starts and ends at I.
                boundary.toggle(I09Monomial::Constant);
            }
            equations.push(boundary.finish());
        }
    }
    if equations
        .iter()
        .flat_map(|equation| equation.iter())
        .any(|term| !monomial_is_in_range(*term, variable_count))
    {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_compiler_emitted_an_out_of_range_variable",
        ));
    }
    Ok(equations)
}

#[derive(Default)]
struct Polynomial(BTreeSet<I09Monomial>);

impl Polynomial {
    fn new() -> Self {
        Self::default()
    }

    fn toggle(&mut self, term: I09Monomial) {
        if !self.0.insert(term) {
            self.0.remove(&term);
        }
    }

    fn finish(self) -> Vec<I09Monomial> {
        self.0.into_iter().collect()
    }
}

fn product(left: usize, right: usize) -> I09Monomial {
    if left == right {
        I09Monomial::Variable(left)
    } else if left < right {
        I09Monomial::Product(left, right)
    } else {
        I09Monomial::Product(right, left)
    }
}

fn checked_index(index: Option<usize>) -> Result<usize, I09Error> {
    index.ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_variable_index_overflow"))
}

fn monomial_is_in_range(term: I09Monomial, variable_count: usize) -> bool {
    match term {
        I09Monomial::Constant => true,
        I09Monomial::Variable(index) => index < variable_count,
        I09Monomial::Product(left, right) => left < variable_count && right < variable_count,
    }
}
