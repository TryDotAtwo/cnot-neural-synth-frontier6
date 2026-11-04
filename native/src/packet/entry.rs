use crate::schema::FeatureSpec;
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

mod validation;

impl FeatureEntry {
    pub fn row_is_available(&self, row: usize) -> bool {
        self.available.get(row).copied() == Some(1)
    }
}
