//! Schema identity, tensor rank, and role-specific shape validation.

use super::super::{FeatureEntry, FeaturePacketError};
use crate::schema::FeatureRole;

pub(super) fn validate(
    entry: &FeatureEntry,
    batch: usize,
    nmax: usize,
) -> Result<(), FeaturePacketError> {
    if entry.spec.channels == 0 {
        return Err(FeaturePacketError(format!(
            "{} must declare at least one channel",
            entry.spec.id
        )));
    }
    if entry.spec.scope.trim().is_empty()
        || entry.spec.producer_version.trim().is_empty()
        || entry.spec.exactness.trim().is_empty()
        || entry.spec.cost_tier.trim().is_empty()
    {
        return Err(FeaturePacketError(format!(
            "{} is missing producer identity metadata",
            entry.spec.id
        )));
    }
    if entry
        .spec
        .dependencies
        .iter()
        .any(|dependency| dependency.trim().is_empty())
    {
        return Err(FeaturePacketError(format!(
            "{} has an empty dependency identity",
            entry.spec.id
        )));
    }
    let rank = match entry.spec.role {
        FeatureRole::Node | FeatureRole::Group => 3,
        FeatureRole::Pair => 4,
        FeatureRole::Global => 2,
    };
    if entry.shape.len() != rank || entry.shape.first().copied() != Some(batch) {
        return Err(FeaturePacketError(format!(
            "{} has an invalid feature shape",
            entry.spec.id
        )));
    }
    if entry.shape.last().copied() != Some(entry.spec.channels) {
        return Err(FeaturePacketError(format!(
            "{} channel count differs from schema",
            entry.spec.id
        )));
    }
    match entry.spec.role {
        FeatureRole::Node if entry.shape[1] != nmax => {
            return Err(FeaturePacketError(format!(
                "{} node width differs from batch",
                entry.spec.id
            )));
        }
        FeatureRole::Pair if entry.shape[1] != nmax || entry.shape[2] != nmax => {
            return Err(FeaturePacketError(format!(
                "{} pair width differs from batch",
                entry.spec.id
            )));
        }
        FeatureRole::Group => {
            if entry.wire_roles != entry.spec.wire_roles
                || entry
                    .spec
                    .wire_roles
                    .iter()
                    .any(|role| role.trim().is_empty())
                || entry
                    .spec
                    .wire_roles
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != entry.spec.wire_roles.len()
            {
                return Err(FeaturePacketError(format!(
                    "{} group role names differ from its schema",
                    entry.spec.id
                )));
            }
            let groups = entry.shape[1];
            let incidence_len = batch
                .checked_mul(groups)
                .and_then(|x| x.checked_mul(nmax))
                .and_then(|x| x.checked_mul(entry.spec.wire_roles.len()))
                .ok_or_else(|| FeaturePacketError("group incidence size overflow".into()))?;
            let token_len = batch
                .checked_mul(groups)
                .ok_or_else(|| FeaturePacketError("group token mask size overflow".into()))?;
            if entry.token_valid.as_ref().map(Vec::len) != Some(token_len)
                || entry.incidence.as_ref().map(Vec::len) != Some(incidence_len)
            {
                return Err(FeaturePacketError(format!(
                    "{} group metadata shape mismatch",
                    entry.spec.id
                )));
            }
        }
        _ => {
            if !entry.spec.wire_roles.is_empty()
                || !entry.wire_roles.is_empty()
                || entry.token_valid.is_some()
                || entry.incidence.is_some()
            {
                return Err(FeaturePacketError(format!(
                    "{} carries group-only metadata",
                    entry.spec.id
                )));
            }
        }
    }
    Ok(())
}
