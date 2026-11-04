//! Neural basis proposal head and its binding to a physical matrix state.

mod loss;
mod proposal;
use super::BasisTemplate;

use crate::model::{FieldOutput, FusionDistanceField, ModelConfig, ModelError};
use crate::packet::FeaturePacket;
use crate::schema::FeatureSpec;
use tch::nn::{self};
use tch::{Device, Tensor};

/// Full-width field plus the paid-basis proposal head used by production runs.
#[derive(Debug)]
pub struct BasisField {
    field: FusionDistanceField,
    proposer: proposal::BasisProposal,
}

#[derive(Debug)]
pub struct BasisFieldOutput {
    pub field: FieldOutput,
    pub proposal_probabilities: Tensor,
}

impl BasisField {
    pub fn new(
        path: &nn::Path<'_>,
        specs: &[FeatureSpec],
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        let field = FusionDistanceField::new(
            &(path / "field"),
            specs,
            schema_id,
            policy_id,
            device,
            config.clone(),
        )?;
        let proposer = proposal::BasisProposal::new(
            &(path / "proposer"),
            config.wire_dim,
            config.basis_proposal_hidden_dim,
        );
        Ok(Self { field, proposer })
    }

    pub fn forward(&self, packet: &FeaturePacket) -> Result<FieldOutput, ModelError> {
        self.field.forward(packet)
    }

    pub fn basis_proposal(
        &self,
        packet: &FeaturePacket,
        template: &BasisTemplate,
        batch_index: usize,
    ) -> Result<Tensor, ModelError> {
        let encoded = self.field.encode(packet)?;
        self.proposer.forward(&encoded, template, batch_index)
    }

    pub fn forward_with_basis(
        &self,
        packet: &FeaturePacket,
        template: &BasisTemplate,
        batch_index: usize,
    ) -> Result<BasisFieldOutput, ModelError> {
        let encoded = self.field.encode(packet)?;
        let field = self.field.forward_encoded(&encoded);
        let proposal_probabilities = self.proposer.forward(&encoded, template, batch_index)?;
        Ok(BasisFieldOutput {
            field,
            proposal_probabilities,
        })
    }
}
