//! Per-slot selector, readout, and matrix-transition equations.

use super::polynomial::{product, Polynomial};
use super::variables::{
    active_var, checked_index, control_var, matrix_var, readout_var, target_var,
};
use crate::i09_quadratic::{I09Context, I09Error, I09Monomial};

pub(super) fn append_slot_equations(
    equations: &mut Vec<Vec<I09Monomial>>,
    slot: usize,
    context: &I09Context,
) -> Result<(), I09Error> {
    let n = context.n;
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
    Ok(())
}
