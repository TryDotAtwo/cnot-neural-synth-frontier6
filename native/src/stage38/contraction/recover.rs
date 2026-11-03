use super::super::{xor_rows, Histories, Stage38Error};
use super::count::count_histories;
use crate::state::CnotGate;
use num_bigint::Sign;
use std::collections::{HashMap, HashSet};

pub(in crate::stage38) fn recover_histories(
    initial: &Histories,
    max_entries: usize,
) -> Result<Vec<CnotGate>, Stage38Error> {
    if count_histories(initial, max_entries, &HashSet::new())?
        .count
        .sign()
        == Sign::NoSign
    {
        return Err(Stage38Error::ContractionInvariant(
            "cannot recover a word from zero temporal amplitude",
        ));
    }
    let protected: HashSet<usize> = initial
        .iter()
        .enumerate()
        .filter_map(|(row, history)| (history.len() > 2).then_some(row))
        .collect();
    let mut histories = initial.clone();
    let mut gates = Vec::new();

    while histories.iter().any(|history| history.len() > 1) {
        let mut lookup = HashMap::new();
        for (row, history) in histories.iter().enumerate() {
            lookup.insert(history[0].clone(), row);
        }
        let mut admitted = false;
        for target in 0..histories.len() {
            if histories[target].len() == 1 {
                continue;
            }
            let difference = xor_rows(&histories[target][0], &histories[target][1]);
            let Some(control) = lookup.get(&difference).copied() else {
                continue;
            };
            if control == target {
                continue;
            }
            let mut updated = histories.clone();
            updated[target].remove(0);
            if count_histories(&updated, max_entries, &protected)?
                .count
                .sign()
                != Sign::Plus
            {
                continue;
            }
            gates.push(CnotGate {
                control: u32::try_from(control).map_err(|_| Stage38Error::SizeOverflow)?,
                target: u32::try_from(target).map_err(|_| Stage38Error::SizeOverflow)?,
            });
            histories = updated;
            admitted = true;
            break;
        }
        if !admitted {
            return Err(Stage38Error::ContractionInvariant(
                "positive amplitude has no recoverable next gate",
            ));
        }
    }
    Ok(gates)
}
