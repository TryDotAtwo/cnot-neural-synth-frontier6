//! Exact small-width paid-basis optimization from the source local atlas.
//!
//! The atlas covers every invertible binary coordinate change on 1..=4 local
//! wires. Its price includes preparation and its physical inverse/restoration,
//! both counted in CNOTs. It optimizes a block's local basis only; it is not a
//! global CNOT shortest-path solver.

use std::fmt;
use std::ops::Range;

use crate::alphabet::WordCertificate;
use crate::state::CnotGate;

mod atlas;
mod compiler;
pub use atlas::optimize_exact_basis;
pub use compiler::compile_paid_shear;
pub(crate) use compiler::{cancel_adjacent_block_gates, endpoint_sha256};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BasisSide {
    Source,
    Target,
}

impl fmt::Display for BasisSide {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Source => "source",
            Self::Target => "target",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaidBasisError(pub String);

impl fmt::Display for PaidBasisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PaidBasisError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactBasisOptimization {
    /// Ordered local add gates `(control, target)` in source atlas order.
    pub operations: Vec<(usize, usize)>,
    /// Hamming weight before changing basis.
    pub direct_cost: usize,
    /// `2 * preparation_gate_count + transformed_core_hamming_weight`.
    pub paid_cost: usize,
    pub transformed_vectors: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaidShearConfig {
    pub local_width: usize,
    pub sweeps: usize,
}

impl PaidShearConfig {
    /// Parameters used by the source paid-basis example compiler.
    pub const fn source_exact() -> Self {
        Self {
            local_width: 4,
            sweeps: 2,
        }
    }
}

impl Default for PaidShearConfig {
    fn default() -> Self {
        Self::source_exact()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedBasisMove {
    pub sweep: usize,
    pub side: BasisSide,
    pub group_indices: Vec<usize>,
    pub cost_before: usize,
    pub cost_after: usize,
    /// Half-open range in the physical preparation word for this local block.
    pub preparation_range: Range<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundaryCancellation {
    pub control: u32,
    pub target: u32,
    pub canceled_pairs: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaidShearCandidate {
    /// Replayable certificate for the complete physical word on all `n` wires.
    pub certificate: WordCertificate,
    /// Physical preparation gates in the source compiler's execution order.
    pub preparation: Vec<CnotGate>,
    pub source_preparation: Vec<CnotGate>,
    pub target_preparation: Vec<CnotGate>,
    pub core: Vec<CnotGate>,
    pub restoration: Vec<CnotGate>,
    pub original_block: Vec<Vec<u8>>,
    pub transformed_block: Vec<Vec<u8>>,
    /// Coordinate changes accumulated by source and target local operations.
    pub source_basis: Vec<Vec<u8>>,
    pub target_basis: Vec<Vec<u8>>,
    pub direct_cost: usize,
    pub paid_cost: usize,
    /// Count of gate pairs removed across adjacent local blocks.
    pub boundary_cancellations: Vec<BoundaryCancellation>,
    pub endpoint_sha256: String,
    pub tested_groups: usize,
    pub accepted_moves: Vec<AcceptedBasisMove>,
}

#[cfg(test)]
mod tests {
    use super::cancel_adjacent_block_gates;
    use crate::state::CnotGate;

    #[test]
    fn adjacent_equal_cnot_blocks_cancel_and_are_audited() {
        let gate = CnotGate {
            control: 2,
            target: 5,
        };
        let audit = cancel_adjacent_block_gates(&[vec![gate], vec![gate]]);
        assert!(audit.word.is_empty());
        assert_eq!(
            audit.boundary_cancellations,
            vec![super::BoundaryCancellation {
                control: 2,
                target: 5,
                canceled_pairs: 1,
            }]
        );
    }
}
