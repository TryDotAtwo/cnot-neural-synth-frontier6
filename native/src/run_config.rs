//! Strict, pinned, bounded CUDA training command.
mod execution;
mod manifest;

pub use execution::execute_run_manifest;
pub use manifest::{parse_run_manifest, RunManifest};
