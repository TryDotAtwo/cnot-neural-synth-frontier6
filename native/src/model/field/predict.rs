use super::FusionDistanceField;
use crate::model::{DeviceFeaturePacket, EncodedState, FieldOutput, ModelError};
use crate::packet::FeaturePacket;
use tch::{Device, Tensor};

impl FusionDistanceField {
    /// Bind live parameter storage to the canonical group activity slots used
    /// by the CUDA optimizer. Create this map after loading the model weights.
    #[cfg(feature = "cuda")]
    pub fn group_parameter_slots(&self) -> std::collections::HashMap<usize, i64> {
        self.stem.group_parameter_slots()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn value_encoded(&self, encoded: &EncodedState, share_wire_projections: bool) -> Tensor {
        let (family_tokens, family_masks) = self.family_heads.forward(encoded);
        let positive_value = self.fusion.forward(
            encoded,
            &family_tokens,
            &family_masks,
            share_wire_projections,
        );
        Tensor::zeros_like(&positive_value).where_self(&encoded.identity, &positive_value)
    }

    pub(crate) fn forward_encoded(&self, encoded: &EncodedState) -> FieldOutput {
        self.forward_encoded_with_projection(encoded, false)
    }

    fn forward_encoded_with_projection(&self, encoded: &EncodedState, share_wire_projections: bool) -> FieldOutput {
        let value = self.value_encoded(encoded, share_wire_projections);
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

    fn encode_device(
        &self,
        packet: &DeviceFeaturePacket,
        share_wire_projections: bool,
    ) -> Result<EncodedState, ModelError> {
        packet.validate_model(
            &self.specs,
            &self.expected_schema_id,
            &self.expected_policy_id,
            self.device,
        )?;
        self.stem.forward(&packet.tensors, share_wire_projections)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn uses_missing_tokens(&self) -> bool {
        self.expected_schema_id == crate::schema::MISSING_TOKEN_LAYOUT_ID
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn value_device(&self, packet: &DeviceFeaturePacket) -> Result<Tensor, ModelError> {
        self.candidate_value_device(packet, false)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn candidate_value_device(
        &self,
        packet: &DeviceFeaturePacket,
        share_wire_projections: bool,
    ) -> Result<Tensor, ModelError> {
        let _phase = crate::stage_profile::stage("model_forward");
        Ok(self.value_encoded(
            &self.encode_device(packet, share_wire_projections)?,
            share_wire_projections,
        ))
    }

    /// The complete field, including heads supervised on the root.
    pub fn forward_device(&self, packet: &DeviceFeaturePacket) -> Result<FieldOutput, ModelError> {
        let _phase = crate::stage_profile::stage("model_forward");
        Ok(self.forward_encoded(&self.encode_device(packet, false)?))
    }

    /// Same complete field with optional factorization for gradient forwards.
    #[cfg(feature = "cuda")]
    pub(crate) fn training_forward_device(
        &self, packet: &DeviceFeaturePacket, share_wire_projections: bool,
    ) -> Result<FieldOutput, ModelError> {
        let _phase = crate::stage_profile::stage("model_forward");
        Ok(self.forward_encoded_with_projection(
            &self.encode_device(packet, share_wire_projections)?, share_wire_projections,
        ))
    }
}
