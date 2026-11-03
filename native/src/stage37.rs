//! Exact recognizer for the zero-excess CNOT sector `A -> I`.
//!
//! For a complete unit-cost CNOT alphabet, every changed target row must be
//! updated at least once. This recognizer checks whether each changed row can
//! reach identity with exactly one update, orders those updates by a DAG, and
//! returns a replayable word. Failure excludes only the changed-row lower
//! bound; it does not estimate the true distance. It is a solver primitive,
//! not an F07 packet provider: F07's native group/incidence contract remains
//! unconfigured.

pub const STAGE37_PRODUCER: &str = "stage37:zero_excess";
pub const STAGE37_SOURCE_NOTES_SHA256: &str =
    "144cfd8d91663581fee9edb79718d5fb93969dcda50a42e8939965024834c6f9";

mod recognition;
mod support;
mod types;

pub use recognition::{recognize_stage37, verify_stage37_result};
pub use types::{
    Stage37Error, Stage37Exact, Stage37NoZeroExcess, Stage37Parent, Stage37ParentMode,
    Stage37Result, Stage37Unsupported,
};
