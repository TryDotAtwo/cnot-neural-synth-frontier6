use crate::block_parities::BlockContext;
use crate::state::MatrixState;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct ParityPattern(pub(super) Vec<u64>);

impl ParityPattern {
    pub(super) fn bit(&self, index: usize) -> bool {
        self.0
            .get(index / 64)
            .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
    }

    pub(super) fn is_zero(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    pub(super) fn intersection_count(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(&other.0)
            .map(|(left, right)| (left & right).count_ones() as usize)
            .sum()
    }
}

impl Ord for ParityPattern {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for ParityPattern {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) struct ParityData {
    pub(super) codes: Vec<ParityPattern>,
    pub(super) counts: HashMap<ParityPattern, usize>,
    pub(super) columns: Vec<ParityPattern>,
}

pub(super) struct PreparedBlock<'a> {
    pub(super) context: &'a BlockContext,
    pub(super) data: ParityData,
    pub(super) patterns: Vec<ParityPattern>,
}

pub(super) fn parity_data(
    state: &MatrixState,
    context: &BlockContext,
    max_patterns: usize,
) -> Result<ParityData, &'static str> {
    let source_count = context.sources.len();
    let target_count = context.targets.len();
    let source_words = source_count
        .checked_add(63)
        .ok_or("source parity width overflow")?
        / 64;
    let target_words = target_count
        .checked_add(63)
        .ok_or("target parity width overflow")?
        / 64;
    let zero = ParityPattern(zeroed_words(source_words)?);
    let mut codes = Vec::new();
    codes
        .try_reserve_exact(target_count)
        .map_err(|_| "target parity allocation failed")?;
    let mut counts = HashMap::new();
    counts
        .try_reserve(
            target_count
                .checked_add(1)
                .ok_or("pattern count overflow")?
                .min(max_patterns),
        )
        .map_err(|_| "pattern count allocation failed")?;
    counts.insert(zero.clone(), 0usize);

    for target in &context.targets {
        let mut words = zeroed_words(source_words)?;
        for (source_index, source) in context.sources.iter().copied().enumerate() {
            if matrix_bit(state, *target, source)? {
                words[source_index / 64] |= 1u64 << (source_index % 64);
            }
        }
        let code = ParityPattern(words);
        if !counts.contains_key(&code) && counts.len() >= max_patterns {
            return Err("distinct parity pattern budget exceeded");
        }
        let count = counts.entry(code.clone()).or_insert(0);
        *count = count.checked_add(1).ok_or("pattern frequency overflow")?;
        codes.push(code);
    }

    let mut columns = Vec::new();
    columns
        .try_reserve_exact(source_count)
        .map_err(|_| "source parity-column allocation failed")?;
    for source_index in 0..source_count {
        let mut words = zeroed_words(target_words)?;
        for (target_index, code) in codes.iter().enumerate() {
            if code.bit(source_index) {
                words[target_index / 64] |= 1u64 << (target_index % 64);
            }
        }
        columns.push(ParityPattern(words));
    }
    Ok(ParityData {
        codes,
        counts,
        columns,
    })
}

fn matrix_bit(state: &MatrixState, row: u32, column: u32) -> Result<bool, &'static str> {
    let index = (row as usize)
        .checked_mul(state.words_per_row())
        .and_then(|base| base.checked_add(column as usize / 64))
        .ok_or("matrix parity index overflow")?;
    let word = state
        .packed_rows()
        .get(index)
        .ok_or("matrix parity index outside packed state")?;
    Ok(word & (1u64 << (column % 64)) != 0)
}

fn zeroed_words(count: usize) -> Result<Vec<u64>, &'static str> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(count)
        .map_err(|_| "parity workspace allocation failed")?;
    words.resize(count, 0);
    Ok(words)
}
