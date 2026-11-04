//! Stable slot-local variable numbering for the I09 quadratic compiler.

use super::super::layout::slot_width;
use crate::i09_quadratic::{I09Error, I09ErrorKind};

pub(super) fn active_var(slot: usize, n: usize) -> Option<usize> {
    slot.checked_mul(slot_width(n)?)
}

pub(super) fn control_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    active_var(slot, n)?.checked_add(1)?.checked_add(wire)
}

pub(super) fn target_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    control_var(slot, n, n)?.checked_add(wire)
}

pub(super) fn readout_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    control_var(slot, n.checked_mul(2)?, n)?.checked_add(wire)
}

pub(super) fn matrix_var(slot: usize, row: usize, column: usize, n: usize) -> Option<usize> {
    let offset = row.checked_mul(n)?.checked_add(column)?;
    readout_var(slot, n, n)?.checked_add(offset)
}

pub(super) fn checked_index(index: Option<usize>) -> Result<usize, I09Error> {
    index.ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_variable_index_overflow"))
}
