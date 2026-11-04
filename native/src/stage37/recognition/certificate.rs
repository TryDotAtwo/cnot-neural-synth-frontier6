use super::super::support::lineage_id;
use super::super::types::{Stage37Error, Stage37Exact, Stage37Parent, Stage37Result};
use super::super::{STAGE37_PRODUCER, STAGE37_SOURCE_NOTES_SHA256};
use super::plan::ExactPlan;
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};

pub(super) fn certify(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    plan: ExactPlan,
) -> Result<Stage37Result, Stage37Error> {
    let ExactPlan {
        identity,
        changed,
        parents,
        order,
        lower_bound,
    } = plan;
    let gate_list = order
        .iter()
        .copied()
        .map(|target| {
            let (control, _) = parents[target].ok_or(Stage37Error::AcyclicityContradiction)?;
            Ok(CnotGate {
                control: u32::try_from(control).map_err(|_| Stage37Error::IndexOverflow)?,
                target: u32::try_from(target).map_err(|_| Stage37Error::IndexOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, Stage37Error>>()?;

    let word = WordCertificate {
        start: state.clone(),
        end: identity,
        gates: gate_list,
        physical_cost: lower_bound,
        producer: STAGE37_PRODUCER.into(),
        lineage_id: lineage_id(state),
    };
    word.verify(alphabet)?;

    let changed_rows = changed
        .iter()
        .copied()
        .map(|row| u32::try_from(row).map_err(|_| Stage37Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let update_order = order
        .iter()
        .copied()
        .map(|row| u32::try_from(row).map_err(|_| Stage37Error::IndexOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let ordered_parents = order
        .iter()
        .copied()
        .map(|target| {
            let (control, mode) = parents[target].ok_or(Stage37Error::AcyclicityContradiction)?;
            Ok(Stage37Parent {
                target_row: u32::try_from(target).map_err(|_| Stage37Error::IndexOverflow)?,
                control_row: u32::try_from(control).map_err(|_| Stage37Error::IndexOverflow)?,
                mode,
            })
        })
        .collect::<Result<Vec<_>, Stage37Error>>()?;

    Ok(Stage37Result::Exact(Box::new(Stage37Exact {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        distance: lower_bound,
        lower_bound,
        changed_rows,
        update_order,
        parents: ordered_parents,
        word,
        source_notes_sha256: STAGE37_SOURCE_NOTES_SHA256,
    })))
}
