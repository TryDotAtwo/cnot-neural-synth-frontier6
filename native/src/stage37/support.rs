use super::types::{Stage37Error, Stage37Result, Stage37Unsupported};
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use sha2::{Digest, Sha256};

pub(super) fn is_complete_unit_alphabet(alphabet: &CnotAlphabet) -> Result<bool, Stage37Error> {
    let n = usize::try_from(alphabet.dimension()).map_err(|_| Stage37Error::IndexOverflow)?;
    let action_count = n
        .checked_mul(n.saturating_sub(1))
        .ok_or(Stage37Error::SizeOverflow)?;
    if alphabet.gates().len() != action_count {
        return Ok(false);
    }
    Ok(alphabet
        .gates()
        .iter()
        .enumerate()
        .all(|(index, gate)| alphabet.cost_at(index) == Some(1) && gate.control != gate.target))
}

pub(super) fn is_identity_row(state: &MatrixState, row: usize) -> bool {
    state.row_words(row as u32).is_some_and(|words| {
        words.iter().enumerate().all(|(word, value)| {
            *value
                == if word == row / 64 {
                    1u64 << (row % 64)
                } else {
                    0
                }
        })
    })
}

pub(super) fn lineage_id(state: &MatrixState) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103-stage37-lineage-v1");
    digest.update(state.dimension().to_le_bytes());
    for word in state.packed_rows() {
        digest.update(word.to_le_bytes());
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("stage37:{}:{hex}", state.dimension())
}

pub(super) fn unsupported(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    reason: &'static str,
) -> Stage37Result {
    Stage37Result::Unsupported(Stage37Unsupported {
        state: state.clone(),
        alphabet_id: alphabet.id().to_owned(),
        reason,
    })
}
