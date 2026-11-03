//! Public model configuration, errors, and forward-output contract.

use crate::packet::FeaturePacketError;
use std::fmt;
use tch::Tensor;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

impl From<FeaturePacketError> for ModelError {
    fn from(error: FeaturePacketError) -> Self {
        Self(error.0)
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub wire_dim: i64,
    pub pair_dim: i64,
    pub structural_blocks: usize,
    pub attention_heads: i64,
    pub feedforward_dim: i64,
    pub fusion_blocks: usize,
    pub basis_proposal_hidden_dim: i64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            wire_dim: 128,
            pair_dim: 32,
            structural_blocks: 6,
            attention_heads: 4,
            feedforward_dim: 256,
            fusion_blocks: 2,
            basis_proposal_hidden_dim: 128,
        }
    }
}

impl ModelConfig {
    pub(super) fn validate(&self) -> Result<(), ModelError> {
        if self.wire_dim < 1
            || self.pair_dim < 1
            || self.attention_heads < 1
            || self.feedforward_dim < 1
            || self.basis_proposal_hidden_dim < 1
            || self.wire_dim % self.attention_heads != 0
        {
            return Err(ModelError(
                "invalid model dimensions or attention head count".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FieldOutput {
    pub value: Tensor,
    pub proposal_logits: Tensor,
    pub qhat: Tensor,
    pub action_mask: Tensor,
}
