//! Feature provider contracts, policy identity, and registry exports.
//!
//! Provider code is deliberately absent until an exact Rust implementation is
//! ported and reviewed. Missing providers are recorded as unavailable; they are
//! never replaced by zero-valued features marked as computed.

mod contexts;
mod error;
mod packet_builder;
mod policy;
mod provider;
mod registry;

pub use contexts::FeatureProviderContexts;
pub use error::FeatureBuildError;
use policy::validate_policy;
pub use policy::FeaturePolicy;
pub use provider::{FeatureComputation, FeatureProducer, FeatureValue};
pub use registry::FeatureRegistry;
