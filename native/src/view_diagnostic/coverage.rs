use super::types::{H07Error, H07InputCoverage, H07InputCoverageEntry};
use crate::packet::{FeaturePacket, FeatureStatus};
use crate::schema::{
    all_model_input_specs, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID, H07_POST_FORWARD_DIAGNOSTIC_ID,
    MODEL_INPUT_COUNT,
};

#[cfg(feature = "torch")]
pub(super) fn h07_input_coverage(
    packet: &FeaturePacket,
    probe_id: &str,
) -> Result<H07InputCoverage, H07Error> {
    if packet.entries.len() != FEATURE_COUNT
        || packet.feature_id_manifest_id != FEATURE_ID_MANIFEST_ID
    {
        return Err(H07Error::ModelInputCoverage {
            probe_id: probe_id.into(),
            reason: "packet does not match the canonical 103-ID manifest".into(),
        });
    }

    let mut entries = Vec::with_capacity(MODEL_INPUT_COUNT);
    for spec in all_model_input_specs() {
        if spec.id == H07_POST_FORWARD_DIAGNOSTIC_ID {
            return Err(H07Error::ModelInputCoverage {
                probe_id: probe_id.into(),
                reason: "H07 was listed as a model input".into(),
            });
        }
        let entry =
            packet
                .entries
                .get(spec.id.index())
                .ok_or_else(|| H07Error::ModelInputCoverage {
                    probe_id: probe_id.into(),
                    reason: format!("{} is missing from the packet", spec.id),
                })?;
        if entry.spec.id != spec.id {
            return Err(H07Error::ModelInputCoverage {
                probe_id: probe_id.into(),
                reason: format!(
                    "packet order mismatch: expected {}, found {}",
                    spec.id, entry.spec.id
                ),
            });
        }
        if entry.status.len() != 1 || entry.available.len() != 1 || entry.reason.len() != 1 {
            return Err(H07Error::ModelInputCoverage {
                probe_id: probe_id.into(),
                reason: format!("{} does not have exactly one metadata row", spec.id),
            });
        }
        let status = entry.status[0];
        let available = entry.available[0];
        let valid_mask = match status {
            FeatureStatus::Computed => available == 1,
            FeatureStatus::Inapplicable
            | FeatureStatus::Unsupported
            | FeatureStatus::ResourceLimit
            | FeatureStatus::NotRequested => available == 0,
        };
        if !valid_mask {
            return Err(H07Error::ModelInputCoverage {
                probe_id: probe_id.into(),
                reason: format!(
                    "{} has inconsistent {} status and availability mask {}",
                    spec.id,
                    status.as_str(),
                    available
                ),
            });
        }
        let reason = entry.reason[0].clone();
        entries.push(H07InputCoverageEntry {
            id: spec.id,
            status,
            available,
            reason,
        });
    }

    Ok(H07InputCoverage {
        schema_id: packet.schema_id.clone(),
        policy_id: packet.policy_id.clone(),
        feature_id_manifest_id: packet.feature_id_manifest_id.into(),
        entries,
    })
}
