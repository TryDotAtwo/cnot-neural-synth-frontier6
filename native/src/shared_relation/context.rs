use std::collections::{BTreeMap, HashSet};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRelationContextError(pub String);

impl fmt::Display for SharedRelationContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SharedRelationContextError {}

/// Explicit physical row vectors and a fixed labelled covector frame.
///
/// `covectors[j]` is packed little-endian in the order of `coordinates`:
/// coordinate 0 is bit 0 of word 0. `rows` and `coordinates` preserve caller
/// order; that order is part of the I06 feature identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRelationContext {
    pub(super) n: u32,
    pub(super) rows: Vec<u32>,
    pub(super) coordinates: Vec<u32>,
    pub(super) covectors: Vec<Vec<u64>>,
}

impl SharedRelationContext {
    pub fn new(
        n: u32,
        rows: Vec<u32>,
        coordinates: Vec<u32>,
        covectors: Vec<Vec<u64>>,
    ) -> Result<Self, SharedRelationContextError> {
        if n == 0 {
            return Err(SharedRelationContextError(
                "positive physical width required".into(),
            ));
        }
        validate_indices(&rows, n, "selected rows")?;
        validate_indices(&coordinates, n, "ambient coordinates")?;
        let coordinate_words = words_for(coordinates.len())
            .ok_or_else(|| SharedRelationContextError("coordinate width overflow".into()))?;
        if covectors.iter().any(|vector| {
            vector.len() != coordinate_words || has_bits_above(vector, coordinates.len())
        }) {
            return Err(SharedRelationContextError(
                "covectors must fit the declared coordinate width".into(),
            ));
        }
        if canonical_basis(&covectors, coordinates.len()).len() != covectors.len() {
            return Err(SharedRelationContextError(
                "covector frame must be linearly independent".into(),
            ));
        }
        Ok(Self {
            n,
            rows,
            coordinates,
            covectors,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    pub fn coordinates(&self) -> &[u32] {
        &self.coordinates
    }

    pub fn covectors(&self) -> &[Vec<u64>] {
        &self.covectors
    }
}

fn validate_indices(
    indices: &[u32],
    n: u32,
    label: &str,
) -> Result<(), SharedRelationContextError> {
    let mut seen = HashSet::with_capacity(indices.len());
    if indices.is_empty()
        || indices
            .iter()
            .any(|index| *index >= n || !seen.insert(*index))
    {
        return Err(SharedRelationContextError(format!(
            "{label} must be nonempty, distinct, and within physical width"
        )));
    }
    Ok(())
}

pub(super) fn words_for(width: usize) -> Option<usize> {
    width.checked_add(63).map(|value| value / 64)
}

pub(super) fn bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

pub(super) fn set_bit(words: &mut [u64], index: usize) {
    words[index / 64] |= 1u64 << (index % 64);
}

fn highest_bit(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().rev().find_map(|(index, word)| {
        (*word != 0).then(|| index * 64 + (63 - word.leading_zeros() as usize))
    })
}

fn has_bits_above(words: &[u64], width: usize) -> bool {
    if width == 0 {
        return words.iter().any(|word| *word != 0);
    }
    let full_words = width / 64;
    if words
        .iter()
        .skip(full_words + usize::from(width % 64 != 0))
        .any(|v| *v != 0)
    {
        return true;
    }
    let tail = width % 64;
    tail != 0
        && words
            .get(full_words)
            .is_some_and(|word| word & !((1u64 << tail) - 1) != 0)
}

/// Canonical reduced basis with highest-set-bit pivots, matching stage-56.
fn canonical_basis(vectors: &[Vec<u64>], width: usize) -> Vec<Vec<u64>> {
    let word_count = words_for(width).unwrap_or_default();
    let mut pivots = BTreeMap::<usize, Vec<u64>>::new();
    for raw in vectors {
        let mut row = vec![0u64; word_count];
        for (target, source) in row.iter_mut().zip(raw) {
            *target = *source;
        }
        for (pivot, basis_row) in pivots.iter().rev() {
            if bit(&row, *pivot) {
                xor_assign(&mut row, basis_row);
            }
        }
        let Some(pivot) = highest_bit(&row) else {
            continue;
        };
        for basis_row in pivots.values_mut() {
            if bit(basis_row, pivot) {
                xor_assign(basis_row, &row);
            }
        }
        pivots.insert(pivot, row);
    }
    pivots.into_iter().rev().map(|(_, row)| row).collect()
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= *b;
    }
}

pub(super) fn nullspace(equations: &[Vec<u64>], width: usize) -> Vec<Vec<u64>> {
    let reduced = canonical_basis(equations, width);
    let pivot_set: HashSet<usize> = reduced.iter().filter_map(|row| highest_bit(row)).collect();
    let mut basis = Vec::new();
    for free in 0..width {
        if pivot_set.contains(&free) {
            continue;
        }
        let mut vector = vec![0u64; words_for(width).unwrap_or_default()];
        set_bit(&mut vector, free);
        for row in &reduced {
            let pivot = highest_bit(row).expect("reduced basis rows are nonzero");
            if bit(row, free) {
                set_bit(&mut vector, pivot);
            }
        }
        basis.push(vector);
    }
    canonical_basis(&basis, width)
}
