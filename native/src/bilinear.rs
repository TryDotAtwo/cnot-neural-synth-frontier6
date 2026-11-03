//! Exact source-defined I05 port reduction for a labelled binary matrix block.
//!
//! The two emitted channels are the rank of the stage-55 residual and the
//! dimension of its free hidden space. They describe a finite matrix-product
//! fiber; they are not CNOT distance or a count of physical circuits.

use std::fmt;

/// SHA-256 of `research_v3/artifacts/frontier61_source_v1/stage55_bilinear_kernel/bilinear_kernel.py`.
pub const STAGE55_SOURCE_SHA256: &str =
    "30c90c396e8f5f1d43921f09a0d07ca723afa6e38a9c0ebed4f68b5b2aab5353";

const DEFAULT_MAX_DIMENSION: usize = 144;
const DEFAULT_MAX_MATRIX_BITS: usize = 100_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BilinearContextError(pub String);

impl fmt::Display for BilinearContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BilinearContextError {}

mod context;
mod producer;

pub use context::BilinearContext;
pub use producer::{register_bilinear_contexts, register_default_bilinear_contexts};
