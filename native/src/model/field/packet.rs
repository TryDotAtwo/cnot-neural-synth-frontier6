use super::FusionDistanceField;
use crate::model::{EncodedState, ModelError};
use crate::packet::FeaturePacket;
use crate::schema::{FeatureId, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID};

impl FusionDistanceField {
    pub(crate) fn encode(&self, packet: &FeaturePacket) -> Result<EncodedState, ModelError> {
        if packet.schema_id != self.expected_schema_id
            || packet.policy_id != self.expected_policy_id
        {
            return Err(ModelError(
                "feature packet schema or policy differs from model contract".into(),
            ));
        }
        if packet.feature_id_manifest_id != FEATURE_ID_MANIFEST_ID
            || packet.entries.len() != FEATURE_COUNT
            || packet
                .entries
                .iter()
                .enumerate()
                .any(|(index, entry)| FeatureId::from_index(index) != Some(entry.spec.id))
        {
            return Err(ModelError(format!(
                "feature packet must preserve the canonical {FEATURE_COUNT}-ID manifest"
            )));
        }
        let model_ids: std::collections::HashSet<_> =
            self.specs.iter().map(|spec| spec.id).collect();
        let packet_ids: std::collections::HashSet<_> = packet
            .entries
            .iter()
            .filter(|entry| {
                entry
                    .status
                    .iter()
                    .any(|status| *status != crate::packet::FeatureStatus::NotRequested)
            })
            .map(|entry| entry.spec.id)
            .collect();
        if model_ids != packet_ids {
            return Err(ModelError(
                "selected packet feature IDs differ from model input schema".into(),
            ));
        }
        if self.specs.iter().any(|spec| {
            packet
                .entries
                .get(spec.id.index())
                .is_none_or(|entry| entry.spec != *spec)
        }) {
            return Err(ModelError(
                "feature packet producer schema differs from model contract".into(),
            ));
        }
        self.stem.forward(packet, self.device)
    }
}
