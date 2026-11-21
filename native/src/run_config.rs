//! Pinned bounded run configuration and CPU feature-profile validation.
#[cfg(feature = "cuda")]
mod execution;
#[cfg(feature = "torch")]
mod hamming;
#[cfg(feature = "torch")]
mod panel;
mod manifest;

#[cfg(feature = "cuda")]
pub use execution::coverage::audit_feature_coverage;
#[cfg(feature = "cuda")]
pub use execution::execute_run_manifest;
#[cfg(feature = "cuda")]
pub use execution::execute_benchmark_manifest;
#[cfg(feature = "cuda")]
pub use execution::h07::execute_dev_h07_manifest;
#[cfg(feature = "cuda")]
pub use execution::search::execute_dev_search_manifest;
#[cfg(feature = "torch")]
pub use hamming::execute_hamming_search_manifest;
pub use manifest::{parse_run_manifest, RunManifest};
