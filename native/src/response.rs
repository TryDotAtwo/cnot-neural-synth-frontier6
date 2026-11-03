//! Exact finite-horizon I10 endpoint response for the full ordered CNOT alphabet.
//!
//! A completed layer counts every physical gate word exactly. If the requested
//! layer exceeds a declared state/transition budget, or an exact coefficient
//! exceeds the native counter width, I10 is unavailable; partial counts are
//! never exposed as exact features.

mod contract;
mod oracle;
mod provider;

pub use contract::{I10Config, I10OracleResult, I10OracleStatus};
pub use oracle::evaluate_i10;
pub use provider::register_tiny_i10;
