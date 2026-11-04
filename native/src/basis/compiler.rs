//! Exact paid-basis compilation into a replayable CNOT word certificate.

use super::BasisTemplate;
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::model::ModelError;
use crate::state::CnotGate;

mod certificate;
mod transform;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasisCandidate {
    pub certificate: WordCertificate,
    pub preparation: Vec<CnotGate>,
    pub core: Vec<CnotGate>,
    pub bits: Vec<bool>,
    pub transformed_block: Vec<Vec<u8>>,
}

pub fn compile_basis(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    bits: &[bool],
) -> Result<BasisCandidate, ModelError> {
    template.validate()?;
    validate_block(template, block)?;
    if bits.len() != template.choices.len() {
        return Err(ModelError(
            "basis assignment width differs from template".into(),
        ));
    }
    let n = u32::try_from(template.n)
        .map_err(|_| ModelError("basis width exceeds CNOT state dimension range".into()))?;
    let alphabet =
        CnotAlphabet::all_to_all_unit(n).map_err(|error| ModelError(error.to_string()))?;
    let prepared = transform::apply(template, block, bits)?;
    certificate::build(template, block, bits, n, &alphabet, prepared)
}

fn validate_block(template: &BasisTemplate, block: &[Vec<u8>]) -> Result<(), ModelError> {
    if block.len() != template.target_wires.len()
        || block
            .iter()
            .any(|row| row.len() != template.source_wires.len() || row.iter().any(|bit| *bit > 1))
    {
        return Err(ModelError(
            "basis block must be a binary target-by-source matrix".into(),
        ));
    }
    Ok(())
}
