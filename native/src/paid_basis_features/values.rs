//! Pack exact paid-basis results into the D09-D14 feature layouts.
mod combined;
mod group_values;

use super::helpers::resource;
use super::PaidBasisBlockContext;
use crate::alphabet::CnotAlphabet;
use crate::features::FeatureValue;
use crate::packet::FeatureStatus;
use crate::paid_basis::PaidShearCandidate;
use crate::state::MatrixState;

#[derive(Clone)]
pub(super) struct PaidBasisFeatureValues {
    pub(super) d09: FeatureValue,
    pub(super) d10: FeatureValue,
    pub(super) d11: FeatureValue,
    pub(super) d12: FeatureValue,
    pub(super) d13: FeatureValue,
    pub(super) d14: FeatureValue,
    pub(super) d14_provenance: Vec<String>,
}

pub(super) fn assemble_feature_values(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    contexts: &[&PaidBasisBlockContext],
    candidates: &[PaidShearCandidate],
) -> Result<PaidBasisFeatureValues, (FeatureStatus, String)> {
    let n = state.dimension() as usize;
    let (d09, d10, d11, d12, d13) = group_values::assemble_group_values(contexts, candidates, n)?;
    let (d14, d14_provenance) =
        combined::assemble_combined_feature_value(state, alphabet, candidates)?;

    Ok(PaidBasisFeatureValues {
        d09,
        d10,
        d11,
        d12,
        d13,
        d14,
        d14_provenance,
    })
}

pub(super) fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> u8 {
    let words = state
        .row_words(row as u32)
        .expect("row index is bounded by the matrix dimension");
    ((words[column / 64] >> (column % 64)) & 1) as u8
}

pub(super) fn exact_count(value: usize) -> Result<f32, (FeatureStatus, String)> {
    if value > (1usize << 24) {
        return Err(resource("paid_basis_integer_exceeds_exact_f32_range"));
    }
    Ok(value as f32)
}
