use super::super::BasisTemplate;
use crate::model::ModelError;

pub(super) fn validate_block(
    template: &BasisTemplate,
    block: &[Vec<u8>],
) -> Result<(), ModelError> {
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
