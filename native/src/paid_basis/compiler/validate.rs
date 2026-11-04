//! Input-contract checks for paid-shear compilation.

use super::super::{PaidBasisError, PaidShearConfig};
use std::collections::HashSet;

pub(super) fn validate_inputs(
    block: &[Vec<u8>],
    dimension: usize,
    source_wires: &[usize],
    target_wires: &[usize],
    config: PaidShearConfig,
) -> Result<u32, PaidBasisError> {
    if dimension == 0 || source_wires.is_empty() || target_wires.is_empty() {
        return Err(PaidBasisError(
            "paid shear requires a positive dimension and two nonempty wire blocks".into(),
        ));
    }
    if !(2..=4).contains(&config.local_width) {
        return Err(PaidBasisError(
            "source exact-local compiler supports local width 2..=4".into(),
        ));
    }
    if block.len() != target_wires.len()
        || block
            .iter()
            .any(|row| row.len() != source_wires.len() || row.iter().any(|bit| *bit > 1))
    {
        return Err(PaidBasisError(
            "paid shear block must be a binary target-by-source matrix".into(),
        ));
    }
    let mut physical_wires = HashSet::with_capacity(source_wires.len() + target_wires.len());
    for &wire in source_wires.iter().chain(target_wires) {
        if wire >= dimension || !physical_wires.insert(wire) {
            return Err(PaidBasisError(
                "paid shear wires must be unique, disjoint, and inside the physical dimension"
                    .into(),
            ));
        }
    }
    u32::try_from(dimension)
        .map_err(|_| PaidBasisError("physical dimension exceeds CNOT index range".into()))
}
