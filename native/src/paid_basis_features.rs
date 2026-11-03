//! T2 paid-basis feature producers D09-D14.
//!
//! Every value is derived from an exact-local candidate whose complete physical
//! CNOT word is replayed before endpoint features are emitted. Endpoint digests
//! remain packet provenance, never model values.

use crate::features::FeatureBuildError;
use helpers::sorted_in_range;

const CACHE_CAPACITY: usize = 16;
const MAX_CONTEXTS: usize = 1024;
const WIRE_ROLES: [&str; 2] = ["source_wire_set", "target_wire_set"];
const D_IDS: [u8; 6] = [9, 10, 11, 12, 13, 14];

mod cache;
mod compute;
mod helpers;
mod registration;
mod values;

pub use registration::register_paid_basis_features;

/// A named physical shear block drawn from the source matrix. Indices must be
/// sorted, unique, in range, and source/target sets must be disjoint.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PaidBasisBlockContext {
    dimension: u32,
    source_wires: Vec<usize>,
    target_wires: Vec<usize>,
}

impl PaidBasisBlockContext {
    pub fn new(
        dimension: u32,
        source_wires: Vec<usize>,
        target_wires: Vec<usize>,
    ) -> Result<Self, FeatureBuildError> {
        if dimension < 2
            || !sorted_in_range(&source_wires, dimension)
            || !sorted_in_range(&target_wires, dimension)
            || source_wires
                .iter()
                .any(|source| target_wires.binary_search(source).is_ok())
        {
            return Err(FeatureBuildError(
                "paid-basis context needs sorted, nonempty, disjoint physical wire sets".into(),
            ));
        }
        Ok(Self {
            dimension,
            source_wires,
            target_wires,
        })
    }
}
