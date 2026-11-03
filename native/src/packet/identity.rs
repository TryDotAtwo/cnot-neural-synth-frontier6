use crate::state::MatrixState;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateIdentity {
    pub dimension: u32,
    pub alphabet_id: String,
    pub packed_rows: Vec<u64>,
}

impl From<&MatrixState> for StateIdentity {
    fn from(state: &MatrixState) -> Self {
        Self {
            dimension: state.dimension(),
            alphabet_id: state.alphabet_id().to_owned(),
            packed_rows: state.packed_rows().to_vec(),
        }
    }
}
