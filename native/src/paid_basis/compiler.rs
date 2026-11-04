//! Deterministic local paid-basis shear compilation and endpoint auditing.

mod candidate;
mod endpoint;
mod groups;
mod matrix;
mod optimize;
mod validate;
mod word;

use super::{PaidBasisError, PaidShearCandidate, PaidShearConfig};

pub(crate) use endpoint::endpoint_sha256;
use optimize::optimize_local_bases;
pub(crate) use word::cancel_adjacent_block_gates;

/// Compiles a binary shear block using deterministic, exact local bases of
/// width at most four. Each accepted move strictly decreases the paid physical
/// count. The resulting certificate covers the complete operator on the given
/// physical wires; this routine makes no global-optimality claim.
pub fn compile_paid_shear(
    block: &[Vec<u8>],
    dimension: usize,
    source_wires: &[usize],
    target_wires: &[usize],
    config: PaidShearConfig,
) -> Result<PaidShearCandidate, PaidBasisError> {
    let dimension_u32 =
        validate::validate_inputs(block, dimension, source_wires, target_wires, config)?;
    let optimization = optimize_local_bases(block, source_wires, target_wires, config)?;
    candidate::build_paid_shear_candidate(
        block,
        dimension_u32,
        source_wires,
        target_wires,
        optimization,
    )
}
