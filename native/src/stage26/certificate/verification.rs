use super::gf2::{low_basis, low_remainder, one_hot, projected, row_value};
use super::Certificate;
use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::BTreeSet;

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
