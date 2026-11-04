//! Numeric storage, row availability, and auxiliary buffer validation.

use super::super::{FeatureEntry, FeaturePacketError, FeatureStatus};

pub(super) fn validate(entry: &FeatureEntry, batch: usize) -> Result<(), FeaturePacketError> {
    let expected_values = entry
        .shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| FeaturePacketError(format!("{} value size overflow", entry.spec.id)))?;
    if entry.values.len() != expected_values {
        return Err(FeaturePacketError(format!(
            "{} value buffer length mismatch",
            entry.spec.id
        )));
    }
    if entry.values.iter().any(|value| !value.is_finite()) {
        return Err(FeaturePacketError(format!(
            "{} contains a non-finite value",
            entry.spec.id
        )));
    }
    if entry.available.len() != batch
        || entry.status.len() != batch
        || entry.reason.len() != batch
        || entry.provenance.len() != batch
    {
        return Err(FeaturePacketError(format!(
            "{} row metadata length mismatch",
            entry.spec.id
        )));
    }
    if entry
        .provenance
        .iter()
        .flatten()
        .any(|item| item.trim().is_empty())
    {
        return Err(FeaturePacketError(format!(
            "{} contains empty provenance metadata",
            entry.spec.id
        )));
    }
    if entry.available.iter().any(|mask| *mask > 1) {
        return Err(FeaturePacketError(format!(
            "{} availability mask is not boolean",
            entry.spec.id
        )));
    }
    for row in 0..batch {
        if entry.available[row] == 1 {
            if entry.status[row] != FeatureStatus::Computed {
                return Err(FeaturePacketError(format!(
                    "{} available row has noncomputed status",
                    entry.spec.id
                )));
            }
        } else if entry.status[row] == FeatureStatus::Computed
            || entry.reason[row].trim().is_empty()
        {
            return Err(FeaturePacketError(format!(
                "{} unavailable row lacks an explicit reason",
                entry.spec.id
            )));
        }
    }
    if let Some(mask) = &entry.token_valid {
        if mask.iter().any(|value| *value > 1) {
            return Err(FeaturePacketError(format!(
                "{} token mask is not boolean",
                entry.spec.id
            )));
        }
    }
    if let Some(incidence) = &entry.incidence {
        if incidence
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(FeaturePacketError(format!(
                "{} incidence must be finite and nonnegative",
                entry.spec.id
            )));
        }
    }
    Ok(())
}
