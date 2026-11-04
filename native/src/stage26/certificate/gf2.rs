use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::BTreeMap;

pub(super) fn one_hot(index: usize) -> BigUint {
    BigUint::from(1u8) << index
}

pub(super) fn row_value(state: &MatrixState, row: usize) -> BigUint {
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

pub(super) fn high_basis(values: impl IntoIterator<Item = BigUint>) -> Vec<BigUint> {
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
pub(super) fn low_basis(values: impl IntoIterator<Item = BigUint>) -> BTreeMap<u64, BigUint> {
    let mut pivots = BTreeMap::<u64, BigUint>::new();
    for mut value in values {
        while let Some(pivot) = lowest_set_bit(&value) {
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

pub(super) fn low_remainder(mut value: BigUint, basis: &BTreeMap<u64, BigUint>) -> BigUint {
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

pub(super) fn projected(value: &BigUint, retained: &[bool]) -> BigUint {
    let mut output = BigUint::from(0u8);
    for (column, keep) in retained.iter().copied().enumerate() {
        if keep && value.bit(column as u64) {
            output |= one_hot(column);
        }
    }
    output
}
