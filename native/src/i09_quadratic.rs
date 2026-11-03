//! Exact, resource-bounded Rust port of the pinned Stage-60 I09 response.
//!
//! This provider emits the compiler equations and the source's fixed-prefix
//! alpha/beta/rank summaries. It does not execute a solver or claim general
//! Frontier61 acceptance.

mod compiler;
mod evaluation;
mod producer;
mod registration;
mod summaries;
mod types;

pub use evaluation::compute_i09_response;
pub use registration::{i09_source_sha256, register_i09_quadratic_response};
pub use types::{
    I09Context, I09ContextError, I09Error, I09ErrorKind, I09Limits, I09Monomial, I09Response,
};

/// SHA-256 of the separately pinned Stage-60 source used by the I09 reference.
pub const I09_STAGE60_SOURCE_SHA256: &str =
    "fcdaec83bff7c111261dfc46c2040ddbbc349dbbc11a712a31af945dca3c6339";

const DEFAULT_MAX_VARIABLES: usize = 128;
const DEFAULT_MAX_EQUATIONS: usize = 512;
const DEFAULT_MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;
