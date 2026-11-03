//! Exact one-excess CNOT recognizer using the source's bounded temporal
//! contraction. Counts use arbitrary-precision integers; resource exhaustion
//! never becomes a negative feature label.

pub const STAGE38_PRODUCER: &str = "stage38_one_excess_exact_v1";
pub const STAGE38_SOURCE_NOTES_SHA256: &str =
    "0815dc96c12052e04d26a04295b26f883c8936fb62d5b4c484b7d3841ee8292d";
pub const STAGE38_SOURCE_IMPLEMENTATION_SHA256: &str =
    "90f4aac082958111293665010a34332f28d5145e957ca2ae49551d199c4cd82e";

type Row = Vec<u64>;
type Histories = Vec<Vec<Row>>;
type Constraint = (usize, usize);

mod contraction;
mod recognition;
mod support;
mod types;

pub use recognition::{recognize_stage38, verify_stage38_result};
use support::xor_rows;
pub use types::{
    Stage38Error, Stage38Exact, Stage38NoOneExcess, Stage38ResourceLimit, Stage38Result,
    Stage38Unsupported,
};
