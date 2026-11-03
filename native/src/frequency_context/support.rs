use super::types::FrequencyContext;
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::state::MatrixState;

pub(super) fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

pub(super) fn has_stage52_interface(
    state: &MatrixState,
    context: &FrequencyContext,
) -> Result<bool, ()> {
    let n = context.n;
    for (index, row) in context.sources.iter().copied().enumerate() {
        for column in 0..n {
            if state_bit(state, row, column) != (column == context.sources[index]) {
                return Ok(false);
            }
        }
    }
    for row in context.internal.iter().copied() {
        if context
            .targets
            .iter()
            .copied()
            .any(|column| state_bit(state, row, column))
        {
            return Ok(false);
        }
    }
    for (row_index, row) in context.targets.iter().copied().enumerate() {
        for (column_index, column) in context.targets.iter().copied().enumerate() {
            if state_bit(state, row, column) != (row_index == column_index) {
                return Ok(false);
            }
        }
    }

    let Some(internal_words) = context.internal.len().checked_add(63).map(|bits| bits / 64) else {
        return Err(());
    };
    let mut b_rows = Vec::new();
    if b_rows.try_reserve_exact(context.internal.len()).is_err() {
        return Err(());
    }
    for row in context.internal.iter().copied() {
        let mut packed = Vec::new();
        if packed.try_reserve_exact(internal_words).is_err() {
            return Err(());
        }
        packed.resize(internal_words, 0u64);
        for (column_index, column) in context.internal.iter().copied().enumerate() {
            if state_bit(state, row, column) {
                packed[column_index / 64] |= 1u64 << (column_index % 64);
            }
        }
        b_rows.push(packed);
    }
    Ok(gf2_rank(&b_rows, context.internal.len()).ok_or(())? == context.internal.len())
}

fn state_bit(state: &MatrixState, row: u32, column: u32) -> bool {
    let word = (column / 64) as usize;
    let bit = 1u64 << (column % 64);
    state
        .row_words(row)
        .and_then(|words| words.get(word))
        .is_some_and(|value| value & bit != 0)
}

pub(super) fn gf2_rank(rows: &[Vec<u64>], columns: usize) -> Option<usize> {
    let word_count = columns.checked_add(63)? / 64;
    if rows.iter().any(|row| row.len() != word_count) {
        return None;
    }
    let mut work = Vec::new();
    work.try_reserve_exact(rows.len()).ok()?;
    for row in rows {
        let mut copy = Vec::new();
        copy.try_reserve_exact(row.len()).ok()?;
        copy.extend_from_slice(row);
        work.push(copy);
    }

    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (rank..work.len()).find(|row| work[*row][word] & bit != 0) else {
            continue;
        };
        work.swap(rank, pivot);
        for row in 0..work.len() {
            if row != rank && work[row][word] & bit != 0 {
                for index in 0..word_count {
                    work[row][index] ^= work[rank][index];
                }
            }
        }
        rank += 1;
        if rank == work.len() {
            break;
        }
    }
    Some(rank)
}
