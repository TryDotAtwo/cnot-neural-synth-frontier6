use super::super::{BellmanError, MacroProvider};
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;

pub(super) struct CandidateSet {
    pub(super) successors: Vec<MatrixState>,
    pub(super) costs: Vec<u64>,
    pub(super) primitive_count: usize,
}

pub(super) fn generate_candidates(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    macro_provider: Option<&dyn MacroProvider>,
) -> Result<CandidateSet, BellmanError> {
    let mut successors = Vec::with_capacity(alphabet.gates().len());
    let mut costs = Vec::with_capacity(alphabet.gates().len());
    for (index, gate) in alphabet.gates().iter().copied().enumerate() {
        let mut successor = state.clone();
        successor
            .apply_cnot(gate)
            .map_err(|error| BellmanError::Invalid(error.to_string()))?;
        successors.push(successor);
        costs.push(u64::from(alphabet.cost_at(index).ok_or_else(|| {
            BellmanError::Invalid("alphabet gate cost is missing".into())
        })?));
    }
    let primitive_count = successors.len();
    if let Some(provider) = macro_provider {
        for certificate in provider.candidates(state).map_err(BellmanError::Invalid)? {
            certificate.verify(alphabet)?;
            if certificate.start != *state || certificate.physical_cost == 0 {
                return Err(BellmanError::Invalid(
                    "macro certificate is not bound to its parent or has nonpositive cost".into(),
                ));
            }
            successors.push(certificate.end.clone());
            costs.push(certificate.physical_cost);
        }
    }
    if successors.is_empty() {
        return Err(BellmanError::Invalid(
            "nonterminal state has an empty move alphabet".into(),
        ));
    }

    Ok(CandidateSet {
        successors,
        costs,
        primitive_count,
    })
}
