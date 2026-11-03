//! Exact stage-35 two-changed-row recognizer from the own-source v13 package.
//!
//! This is a bounded exact oracle, not a general distance estimator. It only
//! returns an exact label when the full unit-cost CNOT alphabet and the
//! at-most-two-changed-row hypotheses hold and the constructed word matches
//! the independent global lower bound.

mod math;
mod recognition;
mod types;

pub use recognition::{recognize_stage35, verify_stage35_result};
pub use types::{
    Stage35Branch, Stage35Error, Stage35Exact, Stage35LowerBoundKind, Stage35Result,
    Stage35Unsupported,
};

pub const STAGE35_THEOREM_SHA256: &str =
    "1b8ef924af6c68af51dd67e0be98148ad7f8e8c272d379ad44943b7baf3fade5";
pub const STAGE35_PRODUCER: &str = "stage35_pair_exact_v1";
