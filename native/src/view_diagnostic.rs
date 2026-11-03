//! H07 post-forward prediction-spread diagnostic.
//!
//! The pinned feature schema keeps H07 unimplemented as a model input and gives
//! it a group/1 placeholder. The normative feature bank describes additional
//! forwards over equivalent views and relabel probes. This module runs those
//! forwards on caller-built packets and emits an auditable diagnostic; probe
//! construction and the exact source availability policy remain explicit.

#[cfg(feature = "torch")]
mod coverage;
#[cfg(feature = "torch")]
mod model;
mod statistics;
mod types;

#[cfg(feature = "torch")]
pub use model::{evaluate_h07_prediction_spread, H07ModelProbe};
pub use statistics::compute_h07_prediction_spread;
pub use types::{
    H07Diagnostic, H07Error, H07InputCoverage, H07InputCoverageEntry, H07ProbeKind,
    H07ProbePrediction,
};

pub const H07_PRODUCER_VERSION: &str = "h07_distance_field_value_population_variance_v3";
