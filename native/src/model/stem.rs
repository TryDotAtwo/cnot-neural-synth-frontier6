use super::StructuralBlock;
use crate::schema::{FeatureId, FeatureSpec};
use std::collections::HashMap;
use tch::{nn, Tensor};

mod forward;
mod initialize;

#[derive(Debug)]
pub(super) struct StructuralStem {
    specs: Vec<FeatureSpec>,
    projections: HashMap<FeatureId, nn::Linear>,
    group_role_projections: HashMap<FeatureId, Vec<nn::Linear>>,
    missing_projection: Option<nn::Linear>,
    raw_matrix: nn::Linear,
    pair_to_wire: nn::Linear,
    size_projection: nn::Linear,
    context_query: nn::Linear,
    context_key: nn::Linear,
    context_value: nn::Linear,
    blocks: Vec<StructuralBlock>,
    wire_dim: i64,
}

#[derive(Debug)]
pub(crate) struct EncodedState {
    pub(crate) wire: Tensor,
    pub(super) pair: Tensor,
    pub(super) context: Tensor,
    pub(super) context_valid: Tensor,
    pub(crate) valid: Tensor,
    pub(super) family_available: HashMap<char, Tensor>,
    pub(super) identity: Tensor,
}

#[cfg(feature = "cuda")]
impl StructuralStem {
    pub(super) fn group_parameter_slots(&self) -> HashMap<usize, i64> {
        let mut groups: Vec<_> = self.group_role_projections.iter().collect();
        groups.sort_by_key(|(id, _)| id.index());
        let mut slots = HashMap::new();
        for (slot, (id, roles)) in groups.into_iter().enumerate() {
            for layer in std::iter::once(&self.projections[id]).chain(roles) {
                for parameter in std::iter::once(&layer.ws).chain(layer.bs.iter()) {
                    slots.insert(parameter.data_ptr() as usize, slot as i64);
                }
            }
        }
        slots
    }
}
