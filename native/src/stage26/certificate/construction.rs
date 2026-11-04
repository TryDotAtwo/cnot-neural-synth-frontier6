use super::gf2::{high_basis, high_remainder, one_hot, projected, row_value};
use super::Certificate;
use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn make_certificate(state: &MatrixState) -> Certificate {
    let n = state.dimension() as usize;
    let rows: Vec<_> = (0..n).map(|row| row_value(state, row)).collect();
    let changed: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(row, value)| *value != one_hot(row))
        .collect();
    let mut off_diagonal_columns = vec![false; n];
    for (row, value) in rows.iter().enumerate() {
        for (column, marked) in off_diagonal_columns.iter_mut().enumerate() {
            if row != column && value.bit(column as u64) {
                *marked = true;
            }
        }
    }
    let retained: Vec<_> = (0..n)
        .map(|wire| !(changed[wire] && !off_diagonal_columns[wire]))
        .collect();
    let changing_retained: Vec<_> = (0..n).map(|wire| changed[wire] && retained[wire]).collect();
    let projected_rows: Vec<_> = rows.iter().map(|row| projected(row, &retained)).collect();

    let mut deleted_columns = BTreeSet::new();
    let mut deletion_witness_rows = BTreeMap::new();
    for (row_index, row) in rows.iter().enumerate() {
        let has_retained_part = retained
            .iter()
            .enumerate()
            .any(|(column, keep)| *keep && row.bit(column as u64));
        if !has_retained_part {
            continue;
        }
        for (column, keep) in retained.iter().enumerate() {
            if !keep && row.bit(column as u64) {
                deleted_columns.insert(column);
                deletion_witness_rows.entry(column).or_insert(row_index);
            }
        }
    }

    let inputs: BTreeSet<_> = retained
        .iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .map(|(wire, _)| one_hot(wire))
        .collect();
    let targets: Vec<_> = projected_rows
        .iter()
        .filter(|value| **value != BigUint::from(0u8) && !inputs.contains(*value))
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let changing_support = rows
        .iter()
        .filter(|value| {
            changing_retained
                .iter()
                .enumerate()
                .any(|(wire, active)| *active && value.bit(wire as u64))
        })
        .map(|value| projected(value, &retained));
    let h_basis = high_basis(changing_support);
    let mut representatives = BTreeMap::<BigUint, BigUint>::new();
    let mut w0_generators = Vec::new();
    for value in inputs
        .iter()
        .chain(&targets)
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        let coset = high_remainder(value.clone(), &h_basis);
        if coset == BigUint::from(0u8) {
            if inputs.contains(&value) {
                w0_generators.push(value);
            }
        } else if let Some(representative) = representatives.get(&coset) {
            w0_generators.push(value ^ representative);
        } else {
            representatives.insert(coset, value);
        }
    }
    let w0_basis = high_basis(w0_generators);
    let targets_in_h: Vec<_> = targets
        .iter()
        .filter(|target| high_remainder((*target).clone(), &h_basis) == BigUint::from(0u8))
        .cloned()
        .collect();
    let mut extended_generators = w0_basis.clone();
    extended_generators.extend(targets_in_h.iter().cloned());
    let extended_basis = high_basis(extended_generators);
    let obstruction_rank = extended_basis.len() - w0_basis.len();
    let lower_bound = deleted_columns.len() + targets.len() + obstruction_rank;
    let certificate = Certificate {
        rows,
        retained,
        projected_rows,
        deleted_columns,
        deletion_witness_rows,
        targets,
        h_dimension: h_basis.len(),
        h_basis,
        w0_dimension: w0_basis.len(),
        obstruction_rank,
        deletions: 0,
        target_count: 0,
        lower_bound,
    };
    Certificate {
        deletions: certificate.deleted_columns.len(),
        target_count: certificate.targets.len(),
        ..certificate
    }
}

pub(super) fn cheap_lower_bound(rows: &[BigUint]) -> usize {
    let n = rows.len();
    let mut defects = Vec::with_capacity(n);
    let mut changed_rows = 0usize;
    let mut changed_columns = vec![false; n];
    for (row, value) in rows.iter().enumerate() {
        let defect = value ^ one_hot(row);
        if defect != BigUint::from(0u8) {
            changed_rows += 1;
        }
        for (column, changed) in changed_columns.iter_mut().enumerate() {
            *changed |= defect.bit(column as u64);
        }
        defects.push(defect);
    }
    let rank_defect = high_basis(defects).len();
    changed_rows
        .max(
            changed_columns
                .into_iter()
                .filter(|changed| *changed)
                .count(),
        )
        .max(rank_defect)
}
