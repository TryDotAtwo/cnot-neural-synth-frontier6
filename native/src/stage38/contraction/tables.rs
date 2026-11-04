use super::super::Stage38Error;
use super::factorization::Factor;
use num_bigint::BigInt;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone)]
struct Table {
    scope: Vec<usize>,
    values: Vec<BigInt>,
}

pub(super) fn lattice_sum(
    q: usize,
    factors: &[Factor],
    order: &[usize],
    max_entries: usize,
) -> Result<BigInt, Stage38Error> {
    let mut tables = factors
        .iter()
        .map(|factor| factor_table(factor, q, max_entries))
        .collect::<Result<Vec<_>, _>>()?;

    for variable in order {
        let mut selected = Vec::new();
        let mut remaining = Vec::new();
        for table in tables.drain(..) {
            if table.scope.binary_search(variable).is_ok() {
                selected.push(table);
            } else {
                remaining.push(table);
            }
        }
        tables = remaining;
        if selected.is_empty() {
            tables.push(Table {
                scope: Vec::new(),
                values: vec![BigInt::from(q as u64)],
            });
            continue;
        }

        let joint: Vec<usize> = selected
            .iter()
            .flat_map(|table| table.scope.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let joint_size = checked_power(q, joint.len(), max_entries)?;
        let axis = joint
            .binary_search(variable)
            .map_err(|_| Stage38Error::ContractionInvariant("elimination variable is absent"))?;
        let mut reduced_scope = joint.clone();
        reduced_scope.remove(axis);
        let reduced_size = checked_power(q, reduced_scope.len(), max_entries)?;
        let mut reduced = zero_vector(reduced_size)?;
        let projections: Vec<Vec<usize>> = selected
            .iter()
            .map(|table| {
                table
                    .scope
                    .iter()
                    .map(|v| {
                        joint.binary_search(v).map_err(|_| {
                            Stage38Error::ContractionInvariant("table scope is outside joint scope")
                        })
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        let mut digits = vec![0usize; joint.len()];
        for flat in 0..joint_size {
            decode_assignment(flat, q, &mut digits);
            let mut product = BigInt::from(1u8);
            for (table, projection) in selected.iter().zip(&projections) {
                let table_index = projection
                    .iter()
                    .fold(0usize, |index, position| index * q + digits[*position]);
                product *= &table.values[table_index];
            }
            let reduced_index = digits
                .iter()
                .enumerate()
                .filter(|(position, _)| *position != axis)
                .fold(0usize, |index, (_, digit)| index * q + digit);
            reduced[reduced_index] += product;
        }
        tables.push(Table {
            scope: reduced_scope,
            values: reduced,
        });
    }

    let mut product = BigInt::from(1u8);
    for table in tables {
        if !table.scope.is_empty() || table.values.len() != 1 {
            return Err(Stage38Error::ContractionInvariant(
                "uncontracted variable remains after elimination",
            ));
        }
        product *= table.values[0].clone();
    }
    Ok(product)
}

fn factor_table(factor: &Factor, q: usize, max_entries: usize) -> Result<Table, Stage38Error> {
    let size = checked_power(q, factor.scope.len(), max_entries)?;
    let mut values = zero_vector(size)?;
    let positions: HashMap<usize, usize> = factor
        .scope
        .iter()
        .copied()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect();
    let mut digits = vec![0usize; factor.scope.len()];
    for (flat, value) in values.iter_mut().enumerate() {
        decode_assignment(flat, q, &mut digits);
        for term in &factor.terms {
            if term
                .iter()
                .all(|(left, right)| digits[positions[left]] < digits[positions[right]])
            {
                *value += BigInt::from(1u8);
            }
        }
    }
    Ok(Table {
        scope: factor.scope.clone(),
        values,
    })
}

pub(super) fn checked_power(
    base: usize,
    exponent: usize,
    limit: usize,
) -> Result<usize, Stage38Error> {
    let mut value = 1usize;
    for _ in 0..exponent {
        value = value
            .checked_mul(base)
            .filter(|value| *value <= limit)
            .ok_or_else(|| {
                Stage38Error::ContractionBudgetExceeded(format!(
                    "table requires more than {limit} entries"
                ))
            })?;
    }
    Ok(value)
}

fn zero_vector(size: usize) -> Result<Vec<BigInt>, Stage38Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| Stage38Error::ContractionBudgetExceeded("table allocation failed".into()))?;
    values.resize_with(size, || BigInt::from(0u8));
    Ok(values)
}

fn decode_assignment(mut flat: usize, radix: usize, digits: &mut [usize]) {
    for digit in digits.iter_mut().rev() {
        *digit = flat % radix;
        flat /= radix;
    }
}
