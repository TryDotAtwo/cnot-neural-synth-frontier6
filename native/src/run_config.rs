//! Strict, pinned, bounded CUDA training command.
mod execution;
mod manifest;

pub use execution::coverage::audit_feature_coverage;
pub use execution::execute_run_manifest;
pub use execution::execute_benchmark_manifest;
pub use execution::h07::execute_dev_h07_manifest;
pub use execution::search::execute_dev_search_manifest;
pub use manifest::{parse_run_manifest, RunManifest};
