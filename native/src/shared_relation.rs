//! Exact I06 port of the pinned stage-56 labelled shared-relation feature.
//!
//! The provider retains the full relation kernel of selected physical rows and
//! each labelled readout through an explicit independent covector frame. It is
//! a boundary signature, not a generic network contraction or CNOT-distance
//! estimate.

/// SHA-256 of the exact stage-56 source used by the Python implementation.
pub const STAGE56_SOURCE_SHA256: &str =
    "b14e85497632860665754a4014c34a4bcf149a9f29a1af6cd3710f401559c7ef";

mod context;
mod producer;
mod registration;

pub use context::{SharedRelationContext, SharedRelationContextError};
pub use registration::{
    register_default_shared_relation_contexts, register_shared_relation_contexts,
};

const DEFAULT_MAX_VECTORS: usize = 32;
const DEFAULT_MAX_COORDINATES: usize = 144;
const DEFAULT_MAX_PORTS: usize = 32;
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const EXACT_F32_INTEGER_LIMIT: usize = 1 << 24;
