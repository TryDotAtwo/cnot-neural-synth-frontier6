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
