//! Exact ports of selected source feature producers.
//!
//! Current built-in coverage is 41 of the canonical 103 IDs: A01-A20,
//! B01-B12, F01, G05, C01, D03 and H01-H05. Contextual providers add
//! coverage only when their explicit physical input contracts are supplied;
//! no placeholder values are emitted for unavailable features.

mod compute;
mod family_a;
mod family_aux;
mod family_b;
mod family_specific;
mod family_views;
mod math;
mod matrix;
mod metadata;
mod output;

pub use compute::compute;
pub(crate) use math::gf2_inverse;
pub use metadata::{is_builtin, metadata, BuiltinMetadata};
