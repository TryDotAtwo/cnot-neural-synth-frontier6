//! Source-compatible guarded recognizer feature views.
//!
//! F07 exposes the bounded stage-37 zero-excess recognizer as three status
//! channels. A negative result excludes only the changed-row lower-bound
//! length; it does not claim that longer CNOT words do not exist.

mod one_excess;
mod pair;
mod zero_excess;

pub use one_excess::register_guarded_one_excess;
pub use pair::{register_guarded_pair, register_guarded_pair_f02_view};
pub use zero_excess::{register_guarded_zero_excess, register_guarded_zero_excess_with_budget};

use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use sha2::{Digest, Sha256};

const DEFAULT_MAX_ENTRIES: usize = 100_000;
const CACHE_CAPACITY: usize = 32;
const STAGE35_SOURCE_SHA256: &str =
    "1b8ef924af6c68af51dd67e0be98148ad7f8e8c272d379ad44943b7baf3fade5";
const STAGE38_SOURCE_NOTES_SHA256: &str =
    "0815dc96c12052e04d26a04295b26f883c8936fb62d5b4c484b7d3841ee8292d";
const STAGE38_SOURCE_IMPLEMENTATION_SHA256: &str =
    "90f4aac082958111293665010a34332f28d5145e957ca2ae49551d199c4cd82e";

fn computed_status(values: Vec<f32>) -> FeatureComputation {
    FeatureComputation::Computed(FeatureValue {
        shape: vec![3],
        values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}

fn guarded_producer_version(max_n: u32, max_entries: usize) -> String {
    // Python's json.dumps tuple encoding is a JSON array with the default
    // separators containing one space after commas. All string fields here
    // are lowercase SHA-256 hex, so this spelling is byte-for-byte stable.
    let source_identity = format!(
        "[{max_n}, {max_entries}, \"{STAGE35_SOURCE_SHA256}\", \"{STAGE38_SOURCE_NOTES_SHA256}\", \"{STAGE38_SOURCE_IMPLEMENTATION_SHA256}\"]"
    );
    let digest = Sha256::digest(source_identity.as_bytes());
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("guarded_certificate_features_v1:{hex}")
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
