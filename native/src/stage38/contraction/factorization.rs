use super::super::{xor_rows, Constraint, Histories, Stage38Error};
use crate::state::MatrixState;
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct Factor {
    pub(super) terms: Vec<Vec<Constraint>>,
    pub(super) scope: Vec<usize>,
}

type HistoryEvent = (usize, usize);
type HistoryFactorization = (Vec<HistoryEvent>, Vec<Factor>);

impl Factor {
    fn new(terms: Vec<Vec<Constraint>>) -> Self {
        let scope = terms
            .iter()
            .flat_map(|term| term.iter().flat_map(|(left, right)| [*left, *right]))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Self { terms, scope }
    }
}

pub(super) fn history_factors(histories: &Histories) -> Result<HistoryFactorization, Stage38Error> {
    validate_histories(histories)?;
    let n = histories.len();
    let mut events = Vec::new();
    let mut labels = HashMap::new();
    for (row, history) in histories.iter().enumerate() {
        for transition in 0..history.len() - 1 {
            let label = events.len();
            labels.insert((row, transition), label);
            events.push((row, transition));
        }
    }

    let mut factors = Vec::new();
    for (row, history) in histories.iter().enumerate() {
        for transition in 0..history.len().saturating_sub(2) {
            factors.push(Factor::new(vec![vec![(
                label(&labels, row, transition)?,
                label(&labels, row, transition + 1)?,
            )]]));
        }
    }

    for (event, (row, transition)) in events.iter().copied().enumerate() {
        let difference = xor_rows(&histories[row][transition], &histories[row][transition + 1]);
        let mut terms = Vec::new();
        for (control_row, history) in histories.iter().enumerate() {
            if control_row == row {
                continue;
            }
            for (position, value) in history.iter().enumerate() {
                if *value != difference {
                    continue;
                }
                let mut term = Vec::new();
                if position > 0 {
                    term.push((label(&labels, control_row, position - 1)?, event));
                }
                if position < history.len() - 1 {
                    term.push((event, label(&labels, control_row, position)?));
                }
                term.sort_unstable();
                terms.push(term);
            }
        }
        factors.push(Factor::new(terms));
    }

    let mut seen = HashSet::new();
    factors.retain(|factor| factor.terms.len() != 1 || seen.insert(factor.clone()));
    if events.len() != labels.len() || n != histories.len() {
        return Err(Stage38Error::ContractionInvariant("event labeling changed"));
    }
    Ok((events, factors))
}

fn validate_histories(histories: &Histories) -> Result<(), Stage38Error> {
    let n = histories.len();
    if n == 0 || histories.iter().any(Vec::is_empty) {
        return Err(Stage38Error::InvalidHistory("empty matrix or row"));
    }
    let dimension = u32::try_from(n).map_err(|_| Stage38Error::SizeOverflow)?;
    let words = n.checked_add(63).ok_or(Stage38Error::SizeOverflow)? / 64;
    for history in histories {
        for value in history {
            if value.len() != words || value.iter().all(|word| *word == 0) {
                return Err(Stage38Error::InvalidHistory("row width or nonzero value"));
            }
            if !n.is_multiple_of(64) && value[words - 1] >> (n % 64) != 0 {
                return Err(Stage38Error::InvalidHistory("row padding bits"));
            }
        }
        if history.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Stage38Error::InvalidHistory(
                "event leaves target unchanged",
            ));
        }
    }
    for endpoint in [0, 1] {
        let rows: Vec<u64> = histories
            .iter()
            .flat_map(|history| {
                if endpoint == 0 {
                    history[0].clone()
                } else {
                    history[history.len() - 1].clone()
                }
            })
            .collect();
        MatrixState::from_packed_rows(dimension, "stage38-history-validation", rows)
            .map_err(|_| Stage38Error::InvalidHistory("singular endpoint matrix"))?;
    }
    Ok(())
}

fn label(
    labels: &HashMap<(usize, usize), usize>,
    row: usize,
    transition: usize,
) -> Result<usize, Stage38Error> {
    labels
        .get(&(row, transition))
        .copied()
        .ok_or(Stage38Error::ContractionInvariant("row event has no label"))
}
