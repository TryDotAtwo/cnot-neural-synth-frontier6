use crate::packet::FeaturePacketError;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureBuildError(pub String);

impl fmt::Display for FeatureBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FeatureBuildError {}

impl From<FeaturePacketError> for FeatureBuildError {
    fn from(error: FeaturePacketError) -> Self {
        Self(error.0)
    }
}
