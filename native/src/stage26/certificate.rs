use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Certificate {
    pub(super) rows: Vec<BigUint>,
    pub(super) retained: Vec<bool>,
    pub(super) projected_rows: Vec<BigUint>,
    pub(super) deleted_columns: BTreeSet<usize>,
    pub(super) deletion_witness_rows: BTreeMap<usize, usize>,
    pub(super) targets: Vec<BigUint>,
    pub(super) h_basis: Vec<BigUint>,
    pub(super) deletions: usize,
    pub(super) target_count: usize,
    pub(super) obstruction_rank: usize,
    pub(super) h_dimension: usize,
    pub(super) w0_dimension: usize,
    pub(super) lower_bound: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum EvaluationError {
    UnsupportedAlphabet,
    ResourceLimit(&'static str),
    VerificationFailed(&'static str),
}

fn one_hot(index: usize) -> BigUint {
    BigUint::from(1u8) << index
}

fn row_value(state: &MatrixState, row: usize) -> BigUint {
    let mut value = BigUint::from(0u8);
    for (word, limb) in state
        .row_words(row as u32)
        .expect("row index is in range")
        .iter()
        .enumerate()
    {
        value |= BigUint::from(*limb) << (word * 64);
    }
    value
}

fn high_basis(values: impl IntoIterator<Item = BigUint>) -> Vec<BigUint> {
    let mut pivots = BTreeMap::<u64, BigUint>::new();
    for mut value in values {
        loop {
            if value == BigUint::from(0u8) {
                break;
            }
            let pivot = value.bits() - 1;
            if let Some(existing) = pivots.get(&pivot) {
                value ^= existing;
            } else {
                pivots.insert(pivot, value);
                break;
            }
        }
    }
    let positions: Vec<_> = pivots.keys().copied().collect();
    for low in &positions {
        for high in &positions {
            if high > low && pivots[high].bit(*low) {
                let low_value = pivots[low].clone();
                *pivots.get_mut(high).expect("pivot was collected") ^= low_value;
            }
        }
    }
    positions
        .into_iter()
        .rev()
        .map(|pivot| pivots.remove(&pivot).expect("pivot was collected"))
        .collect()
}

pub(super) fn high_remainder(mut value: BigUint, basis: &[BigUint]) -> BigUint {
    for vector in basis {
        let pivot = vector.bits() - 1;
        if value.bit(pivot) {
            value ^= vector;
        }
    }
    value
}

fn lowest_set_bit(value: &BigUint) -> Option<u64> {
    (0..value.bits()).find(|bit| value.bit(*bit))
}

// The checker deliberately uses low pivots and pairwise coset differences;
// the constructor uses high pivots and one representative per coset.
fn low_basis(values: impl IntoIterator<Item = BigUint>) -> BTreeMap<u64, BigUint> {
    let mut pivots = BTreeMap::<u64, BigUint>::new();
    for mut value in values {
        loop {
            let Some(pivot) = lowest_set_bit(&value) else {
                break;
            };
            if let Some(existing) = pivots.get(&pivot) {
                value ^= existing;
            } else {
                pivots.insert(pivot, value);
                break;
            }
        }
    }
    pivots
}

fn low_remainder(mut value: BigUint, basis: &BTreeMap<u64, BigUint>) -> BigUint {
    loop {
        let Some(pivot) = lowest_set_bit(&value) else {
            return value;
        };
        let Some(vector) = basis.get(&pivot) else {
            return value;
        };
        value ^= vector;
    }
}

fn projected(value: &BigUint, retained: &[bool]) -> BigUint {
    let mut output = BigUint::from(0u8);
    for (column, keep) in retained.iter().copied().enumerate() {
        if keep && value.bit(column as u64) {
            output |= one_hot(column);
        }
    }
    output
}

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
        .filter_map(|(wire, keep)| keep.then(|| one_hot(wire)))
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

pub(super) fn independently_verify(state: &MatrixState, certificate: &Certificate) -> bool {
    let n = state.dimension() as usize;
    if certificate.rows.len() != n || certificate.retained.len() != n {
        return false;
    }
    let source_rows: Vec<_> = (0..n).map(|row| row_value(state, row)).collect();
    if source_rows != certificate.rows
        || certificate
            .rows
            .iter()
            .map(|row| projected(row, &certificate.retained))
            .collect::<Vec<_>>()
            != certificate.projected_rows
    {
        return false;
    }
    let mut deleted = BTreeSet::new();
    for (row_index, row) in certificate.rows.iter().enumerate() {
        let has_retained_part = certificate
            .retained
            .iter()
            .enumerate()
            .any(|(column, keep)| *keep && row.bit(column as u64));
        if has_retained_part {
            for (column, keep) in certificate.retained.iter().enumerate() {
                if !keep && row.bit(column as u64) {
                    deleted.insert(column);
                    if certificate.deletion_witness_rows.get(&column) != Some(&row_index) {
                        // A valid witness may be any row witnessing this column.
                        if !certificate
                            .deletion_witness_rows
                            .get(&column)
                            .is_some_and(|witness| {
                                *witness < n
                                    && certificate.rows[*witness].bit(column as u64)
                                    && certificate.retained.iter().enumerate().any(
                                        |(wire, keep)| {
                                            *keep && certificate.rows[*witness].bit(wire as u64)
                                        },
                                    )
                            })
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    if deleted != certificate.deleted_columns || deleted.len() != certificate.deletions {
        return false;
    }
    if certificate
        .deletion_witness_rows
        .keys()
        .copied()
        .collect::<BTreeSet<_>>()
        != deleted
    {
        return false;
    }

    let inputs: BTreeSet<_> = certificate
        .retained
        .iter()
        .enumerate()
        .filter_map(|(wire, keep)| keep.then(|| one_hot(wire)))
        .collect();
    let targets: BTreeSet<_> = certificate
        .projected_rows
        .iter()
        .filter(|value| **value != BigUint::from(0u8) && !inputs.contains(*value))
        .cloned()
        .collect();
    if targets.len() != certificate.target_count
        || targets.into_iter().collect::<Vec<_>>() != certificate.targets
    {
        return false;
    }

    let h_basis = low_basis(certificate.h_basis.iter().cloned());
    if h_basis.len() != certificate.h_dimension {
        return false;
    }
    for vector in &certificate.h_basis {
        if (0..n).any(|wire| !certificate.retained[wire] && vector.bit(wire as u64)) {
            return false;
        }
    }
    let in_h = |value: &BigUint| low_remainder(value.clone(), &h_basis) == BigUint::from(0u8);

    let mut pool = inputs.clone();
    pool.extend(certificate.targets.iter().cloned());
    let outside: Vec<_> = pool.into_iter().filter(|value| !in_h(value)).collect();
    let mut w0 = Vec::new();
    w0.extend(inputs.iter().filter(|value| in_h(value)).cloned());
    for left in 0..outside.len() {
        for right in left + 1..outside.len() {
            let difference = &outside[left] ^ &outside[right];
            if in_h(&difference) {
                w0.push(difference);
            }
        }
    }
    let w0_basis = low_basis(w0);
    let mut extended: Vec<_> = w0_basis.values().cloned().collect();
    extended.extend(
        certificate
            .targets
            .iter()
            .filter(|value| in_h(value))
            .cloned(),
    );
    let extended_basis = low_basis(extended);
    let obstruction = extended_basis.len() - w0_basis.len();
    if obstruction != certificate.obstruction_rank || w0_basis.len() != certificate.w0_dimension {
        return false;
    }
    certificate.lower_bound == certificate.deletions + certificate.target_count + obstruction
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
