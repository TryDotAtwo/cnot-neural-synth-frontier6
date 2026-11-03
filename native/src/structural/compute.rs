use crate::features::FeatureValue;
use crate::schema::FeatureId;
use crate::state::MatrixState;
use std::collections::HashMap;

use super::matrix::StructuralMatrix;

pub fn compute(
    state: &MatrixState,
    enabled_ids: &[FeatureId],
) -> Result<HashMap<FeatureId, FeatureValue>, String> {
    let data = StructuralMatrix::from_state(state)?;
    let mut output = HashMap::with_capacity(41);
    super::family_a::compute(&data, &mut output)?;
    super::family_b::compute(&data, &mut output)?;
    super::family_aux::compute(&data, enabled_ids, &mut output)?;
    super::family_specific::compute(state, &data, enabled_ids, &mut output)?;
    super::family_views::compute(state, &data, enabled_ids, &mut output)?;
    Ok(output)
}
