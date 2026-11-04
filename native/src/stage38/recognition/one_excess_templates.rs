//! Enumeration and exact contraction audit of one-excess row histories.

use crate::state::MatrixState;
use num_bigint::BigInt;
use std::collections::HashSet;

use super::super::contraction::count_histories;
use super::super::support::{numeric_row_cmp, zero_row};
use super::super::types::Stage38Error;
use super::super::{xor_rows, Histories};

pub(super) struct Enumeration {
    pub(super) total: BigInt,
    pub(super) templates: usize,
    pub(super) max_bag: usize,
    pub(super) chosen: Option<Histories>,
}

pub(super) fn enumerate(
    state: &MatrixState,
    max_entries: usize,
    initial: &[Vec<u64>],
    final_rows: &[Vec<u64>],
    changed: &[usize],
) -> Result<Enumeration, Stage38Error> {
    let n = initial.len();
    let mut total = BigInt::from(0u8);
    let mut templates = 0usize;
    let mut max_bag = 0usize;
    let mut chosen: Option<Histories> = None;

    for target in changed.iter().copied() {
        let mut endpoints = HashSet::new();
        for row in 0..n {
            if row == target {
                continue;
            }
            endpoints.insert(initial[row].clone());
            endpoints.insert(final_rows[row].clone());
        }

        let mut candidates = HashSet::new();
        for endpoint in &endpoints {
            let intermediate = xor_rows(&initial[target], endpoint);
            if intermediate != zero_row(state.words_per_row())
                && intermediate != initial[target]
                && intermediate != final_rows[target]
            {
                candidates.insert(intermediate);
            }
        }
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_by(numeric_row_cmp);

        for intermediate in candidates {
            if !endpoints.contains(&xor_rows(&intermediate, &final_rows[target])) {
                continue;
            }
            let histories: Histories = (0..n)
                .map(|row| {
                    if row == target {
                        vec![
                            initial[row].clone(),
                            intermediate.clone(),
                            final_rows[row].clone(),
                        ]
                    } else if initial[row] != final_rows[row] {
                        vec![initial[row].clone(), final_rows[row].clone()]
                    } else {
                        vec![initial[row].clone()]
                    }
                })
                .collect();
            templates = templates.checked_add(1).ok_or(Stage38Error::SizeOverflow)?;
            let audit = count_histories(&histories, max_entries, &HashSet::new())?;
            max_bag = max_bag.max(audit.max_bag);
            total += &audit.count;
            if audit.count.sign() == num_bigint::Sign::Plus && chosen.is_none() {
                chosen = Some(histories);
            }
        }
    }
    Ok(Enumeration {
        total,
        templates,
        max_bag,
        chosen,
    })
}
