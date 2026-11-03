use crate::schema::{FeatureRole, FeatureSpec};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureStatus {
    Computed,
    Inapplicable,
    Unsupported,
    ResourceLimit,
    NotRequested,
}

impl FeatureStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Computed => "computed",
            Self::Inapplicable => "inapplicable",
            Self::Unsupported => "unsupported",
            Self::ResourceLimit => "resource_limit",
            Self::NotRequested => "not_requested",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePacketError(pub String);

impl fmt::Display for FeaturePacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FeaturePacketError {}

/// Dense values are storage only: an unavailable row is invisible to consumers
/// unless its availability byte is set. The zero-filled row is never evidence.
#[derive(Clone, Debug)]
pub struct FeatureEntry {
    pub spec: FeatureSpec,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub available: Vec<u8>,
    pub status: Vec<FeatureStatus>,
    pub reason: Vec<String>,
    /// Per-row audit values; kept outside the numeric tensor used by models.
    pub provenance: Vec<Vec<String>>,
    pub wire_roles: Vec<String>,
    pub token_valid: Option<Vec<u8>>,
    pub incidence: Option<Vec<f32>>,
}

impl FeatureEntry {
    pub fn validate(&self, batch: usize, nmax: usize) -> Result<(), FeaturePacketError> {
        if self.spec.channels == 0 {
            return Err(FeaturePacketError(format!(
                "{} must declare at least one channel",
                self.spec.id
            )));
        }
        if self.spec.scope.trim().is_empty()
            || self.spec.producer_version.trim().is_empty()
            || self.spec.exactness.trim().is_empty()
            || self.spec.cost_tier.trim().is_empty()
        {
            return Err(FeaturePacketError(format!(
                "{} is missing producer identity metadata",
                self.spec.id
            )));
        }
        if self
            .spec
            .dependencies
            .iter()
            .any(|dependency| dependency.trim().is_empty())
        {
            return Err(FeaturePacketError(format!(
                "{} has an empty dependency identity",
                self.spec.id
            )));
        }
        let rank = match self.spec.role {
            FeatureRole::Node | FeatureRole::Group => 3,
            FeatureRole::Pair => 4,
            FeatureRole::Global => 2,
        };
        if self.shape.len() != rank || self.shape.first().copied() != Some(batch) {
            return Err(FeaturePacketError(format!(
                "{} has an invalid feature shape",
                self.spec.id
            )));
        }
        if self.shape.last().copied() != Some(self.spec.channels) {
            return Err(FeaturePacketError(format!(
                "{} channel count differs from schema",
                self.spec.id
            )));
        }
        match self.spec.role {
            FeatureRole::Node if self.shape[1] != nmax => {
                return Err(FeaturePacketError(format!(
                    "{} node width differs from batch",
                    self.spec.id
                )));
            }
            FeatureRole::Pair if self.shape[1] != nmax || self.shape[2] != nmax => {
                return Err(FeaturePacketError(format!(
                    "{} pair width differs from batch",
                    self.spec.id
                )));
            }
            FeatureRole::Group => {
                if self.wire_roles != self.spec.wire_roles
                    || self
                        .spec
                        .wire_roles
                        .iter()
                        .any(|role| role.trim().is_empty())
                    || self
                        .spec
                        .wire_roles
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != self.spec.wire_roles.len()
                {
                    return Err(FeaturePacketError(format!(
                        "{} group role names differ from its schema",
                        self.spec.id
                    )));
                }
                let groups = self.shape[1];
                let incidence_len = batch
                    .checked_mul(groups)
                    .and_then(|x| x.checked_mul(nmax))
                    .and_then(|x| x.checked_mul(self.spec.wire_roles.len()))
                    .ok_or_else(|| FeaturePacketError("group incidence size overflow".into()))?;
                let token_len = batch
                    .checked_mul(groups)
                    .ok_or_else(|| FeaturePacketError("group token mask size overflow".into()))?;
                if self.token_valid.as_ref().map(Vec::len) != Some(token_len)
                    || self.incidence.as_ref().map(Vec::len) != Some(incidence_len)
                {
                    return Err(FeaturePacketError(format!(
                        "{} group metadata shape mismatch",
                        self.spec.id
                    )));
                }
            }
            _ => {
                if !self.spec.wire_roles.is_empty()
                    || !self.wire_roles.is_empty()
                    || self.token_valid.is_some()
                    || self.incidence.is_some()
                {
                    return Err(FeaturePacketError(format!(
                        "{} carries group-only metadata",
                        self.spec.id
                    )));
                }
            }
        }
        let expected_values = self
            .shape
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .ok_or_else(|| FeaturePacketError(format!("{} value size overflow", self.spec.id)))?;
        if self.values.len() != expected_values {
            return Err(FeaturePacketError(format!(
                "{} value buffer length mismatch",
                self.spec.id
            )));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(FeaturePacketError(format!(
                "{} contains a non-finite value",
                self.spec.id
            )));
        }
        if self.available.len() != batch
            || self.status.len() != batch
            || self.reason.len() != batch
            || self.provenance.len() != batch
        {
            return Err(FeaturePacketError(format!(
                "{} row metadata length mismatch",
                self.spec.id
            )));
        }
        if self
            .provenance
            .iter()
            .flatten()
            .any(|item| item.trim().is_empty())
        {
            return Err(FeaturePacketError(format!(
                "{} contains empty provenance metadata",
                self.spec.id
            )));
        }
        if self.available.iter().any(|mask| *mask > 1) {
            return Err(FeaturePacketError(format!(
                "{} availability mask is not boolean",
                self.spec.id
            )));
        }
        for row in 0..batch {
            if self.available[row] == 1 {
                if self.status[row] != FeatureStatus::Computed {
                    return Err(FeaturePacketError(format!(
                        "{} available row has noncomputed status",
                        self.spec.id
                    )));
                }
            } else if self.status[row] == FeatureStatus::Computed
                || self.reason[row].trim().is_empty()
            {
                return Err(FeaturePacketError(format!(
                    "{} unavailable row lacks an explicit reason",
                    self.spec.id
                )));
            }
        }
        if let Some(mask) = &self.token_valid {
            if mask.iter().any(|value| *value > 1) {
                return Err(FeaturePacketError(format!(
                    "{} token mask is not boolean",
                    self.spec.id
                )));
            }
        }
        if let Some(incidence) = &self.incidence {
            if incidence
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(FeaturePacketError(format!(
                    "{} incidence must be finite and nonnegative",
                    self.spec.id
                )));
            }
        }
        Ok(())
    }

    pub fn row_is_available(&self, row: usize) -> bool {
        self.available.get(row).copied() == Some(1)
    }
}
