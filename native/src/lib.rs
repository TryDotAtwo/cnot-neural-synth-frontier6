#![deny(unsafe_op_in_unsafe_fn)]

pub mod alphabet;
pub mod features;
pub mod packet;
pub mod schema;
pub mod state;
pub mod structural;

#[cfg(feature = "torch")]
pub mod model;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use alphabet::{AlphabetError, CertificateError, CnotAlphabet, WordCertificate};
pub use features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
pub use packet::{FeatureEntry, FeaturePacket, FeaturePacketError, FeatureStatus, StateIdentity};
pub use schema::{all_feature_specs, FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT};
pub use state::{CnotGate, MatrixState};
