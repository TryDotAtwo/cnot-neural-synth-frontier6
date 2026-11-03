//! Checked source port of the stage-26 shared-parity lower-bound features.
//!
//! F03 reports certificate dimensions, F04 reports projected coset relations,
//! and F05 reports the certified lower bound next to the elementary bound.
//! These are lower-bound diagnostics, never exact CNOT distances.

mod certificate;
mod evaluation;
mod producer;
mod registration;

pub const STAGE26_CERTIFICATE_SOURCE_SHA256: &str =
    "6d29d12d2eb369e120533e1961f4b984429643a08f93cc4db99b148a9088bc05";
pub const STAGE26_CHECKER_SOURCE_SHA256: &str =
    "4dfda79863fe7a7b57785e023471cc5d0e8dac21cc3313271cacf4bf97db08ce";
pub const STAGE26_PROOF_NOTES_SHA256: &str =
    "0daadfb3738478b6ee95c11aad4d67177a1c17613eba5e3c91dc041151676fea";
const DEFAULT_MAX_N: u32 = 144;
const DEFAULT_MAX_CHECKER_PAIRS: usize = 100_000;
const CACHE_CAPACITY: usize = 32;
const STAGE26_IDS: [(u8, u8, usize); 3] = [(b'F', 3, 5), (b'F', 4, 3), (b'F', 5, 2)];

pub use registration::{register_stage26_features, register_stage26_features_with_budget};
