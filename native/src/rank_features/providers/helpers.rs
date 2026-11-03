//! Small shared registration and unavailable-result helpers.
use crate::features::{FeatureBuildError, FeatureComputation, FeatureRegistry};
use crate::packet::FeatureStatus;
use crate::schema::FeatureId;

pub(super) fn ensure_unconfigured(
    registry: &FeatureRegistry,
    ids: &[FeatureId],
) -> Result<(), FeatureBuildError> {
    let specs = registry.specs();
    for id in ids {
        if specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }
    Ok(())
}

pub(super) fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
