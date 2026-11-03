use super::FusionDistanceField;
use crate::model::{EncodedState, FieldOutput, ModelError};
use crate::packet::FeaturePacket;
use tch::{Device, Tensor};

impl FusionDistanceField {
    pub fn device(&self) -> Device {
        self.device
    }

    pub(crate) fn forward_encoded(&self, encoded: &EncodedState) -> FieldOutput {
        let (family_tokens, family_masks) = self.family_heads.forward(&encoded);
        let positive_value = self.fusion.forward(&encoded, &family_tokens, &family_masks);
        let value =
            Tensor::zeros_like(&positive_value).where_self(&encoded.identity, &positive_value);
        let (proposal_logits, qhat, action_mask) =
            self.auxiliary
                .forward(&encoded.wire, &encoded.pair, &value, &encoded.valid);
        FieldOutput {
            value,
            proposal_logits,
            qhat,
            action_mask,
        }
    }

    pub fn forward(&self, packet: &FeaturePacket) -> Result<FieldOutput, ModelError> {
        let encoded = self.encode(packet)?;
        Ok(self.forward_encoded(&encoded))
    }
}
