//! Source-relative I07 rank of an explicitly labelled stage-52 interface.
//!
//! The Fourier frequency is supplied by the caller. This provider reports
//! `kappa` and the exact representation-size exponent `w*kappa`; it does not
//! enumerate a spectrum or infer a frequency from the matrix rank.

const STAGE52_PRIMARY_SHA256: &str =
    "44fa3c82aa7b5c59542cb4101c40b8f437fca1e7baf47f66a3c395627c49c465";
const DEFAULT_MAX_WIDTH: u32 = 144;
const DEFAULT_MAX_FREQUENCY_BITS: usize = 20_736;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

mod fingerprint;
mod producer;
mod support;
mod types;

pub use producer::{register_default_frequency_contexts, register_frequency_contexts};
pub use types::{FrequencyContext, FrequencyContextError};
