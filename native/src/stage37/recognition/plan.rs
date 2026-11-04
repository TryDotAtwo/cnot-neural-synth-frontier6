use super::super::support::{is_complete_unit_alphabet, is_identity_row, unsupported};
use super::super::types::{Stage37Error, Stage37NoZeroExcess, Stage37ParentMode, Stage37Result};
use super::super::STAGE37_SOURCE_NOTES_SHA256;
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use std::collections::{BTreeSet, HashMap, VecDeque};

pub(super) enum RecognitionPlan {
    Complete(Stage37Result),
    Exact(ExactPlan),
}

pub(super) struct ExactPlan {
    pub(super) identity: MatrixState,
    pub(super) changed: Vec<usize>,
    pub(super) parents: Vec<Option<(usize, Stage37ParentMode)>>,
    pub(super) order: Vec<usize>,
    pub(super) lower_bound: u64,
}

pub(super) fn analyze(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
) -> Result<RecognitionPlan, Stage37Error> {
    if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id() {
        return Err(Stage37Error::AlphabetMismatch);
    }
    if !is_complete_unit_alphabet(alphabet)? {
        return Ok(RecognitionPlan::Complete(unsupported(
            state,
            alphabet,
            "requires complete unit-cost all-to-all CNOT alphabet",
        )));
    }

    let dimension = usize::try_from(state.dimension()).map_err(|_| Stage37Error::IndexOverflow)?;
    let identity = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|_| Stage37Error::AlphabetMismatch)?;
    let changed: Vec<usize> = (0..dimension)
        .filter(|row| !is_identity_row(state, *row))
        .collect();
    let mut changed_set = vec![false; dimension];
    for row in changed.iter().copied() {
        changed_set[row] = true;
    }

    let mut initial_rows = HashMap::with_capacity(dimension);
    for row in 0..dimension {
        initial_rows.insert(state.row_words(row as u32).unwrap().to_vec(), row);
    }
    let mut final_rows = HashMap::with_capacity(dimension);
    for row in 0..dimension {
        let mut value = vec![0u64; state.words_per_row()];
        value[row / 64] = 1u64 << (row % 64);
        final_rows.insert(value, row);
    }

    let mut parents = vec![None; dimension];
    let mut edges = vec![BTreeSet::new(); dimension];
    let mut indegree = vec![0usize; dimension];
    let mut missing = Vec::new();

    for target in changed.iter().copied() {
        let mut difference = state.row_words(target as u32).unwrap().to_vec();
        difference[target / 64] ^= 1u64 << (target % 64);

        let initial_parent = initial_rows
            .get(&difference)
            .copied()
            .filter(|parent| *parent != target);
        let parent = if let Some(control) = initial_parent {
            Some((control, Stage37ParentMode::InitialRow))
        } else {
            final_rows
                .get(&difference)
                .copied()
                .filter(|parent| *parent != target)
                .map(|control| (control, Stage37ParentMode::FinalRow))
        };

        let Some((control, mode)) = parent else {
            missing.push(target);
            continue;
        };
        parents[target] = Some((control, mode));

        if changed_set[control] {
            let (source, destination) = match mode {
                Stage37ParentMode::InitialRow => (target, control),
                Stage37ParentMode::FinalRow => (control, target),
            };
            if edges[source].insert(destination) {
                indegree[destination] += 1;
            }
        }
    }

    let lower_bound = u64::try_from(changed.len()).map_err(|_| Stage37Error::SizeOverflow)?;
    if !missing.is_empty() {
        let missing_rows = missing
            .into_iter()
            .map(|row| u32::try_from(row).map_err(|_| Stage37Error::IndexOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(RecognitionPlan::Complete(Stage37Result::NoZeroExcess(
            Box::new(Stage37NoZeroExcess {
                state: state.clone(),
                alphabet_id: alphabet.id().to_owned(),
                excluded_length: lower_bound,
                missing_rows,
                source_notes_sha256: STAGE37_SOURCE_NOTES_SHA256,
            }),
        )));
    }

    let mut queue = VecDeque::new();
    for row in changed.iter().copied() {
        if indegree[row] == 0 {
            queue.push_back(row);
        }
    }
    let mut order = Vec::with_capacity(changed.len());
    while let Some(row) = queue.pop_front() {
        order.push(row);
        for dependent in edges[row].iter().copied() {
            indegree[dependent] -= 1;
            if indegree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }
    if order.len() != changed.len() {
        return Err(Stage37Error::AcyclicityContradiction);
    }

    Ok(RecognitionPlan::Exact(ExactPlan {
        identity,
        changed,
        parents,
        order,
        lower_bound,
    }))
}
