//! Deterministic local-basis search and accepted-move accounting.

use super::super::{
    optimize_exact_basis, AcceptedBasisMove, BasisSide, PaidBasisError, PaidShearConfig,
};
use super::groups::{local_vectors, select_groups, side_vectors};
use super::matrix::{identity_matrix, paid_block_cost, physical_wire, xor_rows};
use crate::state::CnotGate;

pub(super) struct OptimizationResult {
    pub(super) transformed_block: Vec<Vec<u8>>,
    pub(super) source_basis: Vec<Vec<u8>>,
    pub(super) target_basis: Vec<Vec<u8>>,
    pub(super) preparation: Vec<CnotGate>,
    pub(super) source_preparation: Vec<CnotGate>,
    pub(super) target_preparation: Vec<CnotGate>,
    pub(super) tested_groups: usize,
    pub(super) accepted_moves: Vec<AcceptedBasisMove>,
}

pub(super) fn optimize_local_bases(
    block: &[Vec<u8>],
    source_wires: &[usize],
    target_wires: &[usize],
    config: PaidShearConfig,
) -> Result<OptimizationResult, PaidBasisError> {
    let mut transformed_block = block.to_vec();
    let mut source_basis = identity_matrix(source_wires.len());
    let mut target_basis = identity_matrix(target_wires.len());
    let mut preparation = Vec::new();
    let mut source_preparation = Vec::new();
    let mut target_preparation = Vec::new();
    let mut tested_groups = 0usize;
    let mut accepted_moves = Vec::new();

    for sweep in 0..config.sweeps {
        let sweep_cost_before = paid_block_cost(&preparation, &transformed_block)?;
        for side in [BasisSide::Source, BasisSide::Target] {
            let vectors = side_vectors(&transformed_block, side);
            for group_indices in select_groups(&vectors, config.local_width) {
                tested_groups = tested_groups
                    .checked_add(1)
                    .ok_or_else(|| PaidBasisError("tested group counter overflow".into()))?;
                let candidate_vectors = local_vectors(&transformed_block, &group_indices, side);
                let optimized = optimize_exact_basis(&candidate_vectors, side)?;
                if optimized.operations.is_empty() {
                    continue;
                }

                let cost_before = paid_block_cost(&preparation, &transformed_block)?;
                let preparation_start = preparation.len();
                for &(local_control, local_target) in &optimized.operations {
                    let control = group_indices[local_control];
                    let target = group_indices[local_target];
                    match side {
                        BasisSide::Source => {
                            for row in &mut transformed_block {
                                row[control] ^= row[target];
                            }
                            xor_rows(&mut source_basis, target, control);
                            let gate = CnotGate {
                                control: physical_wire(source_wires[control])?,
                                target: physical_wire(source_wires[target])?,
                            };
                            preparation.push(gate);
                            source_preparation.push(gate);
                        }
                        BasisSide::Target => {
                            let source_row = transformed_block[control].clone();
                            for (value, source) in
                                transformed_block[target].iter_mut().zip(source_row)
                            {
                                *value ^= source;
                            }
                            xor_rows(&mut target_basis, target, control);
                            let gate = CnotGate {
                                control: physical_wire(target_wires[control])?,
                                target: physical_wire(target_wires[target])?,
                            };
                            preparation.push(gate);
                            target_preparation.push(gate);
                        }
                    }
                }

                let actual = local_vectors(&transformed_block, &group_indices, side);
                if actual != optimized.transformed_vectors {
                    return Err(PaidBasisError(
                        "local basis lift differs from its exact atlas result".into(),
                    ));
                }
                let cost_after = paid_block_cost(&preparation, &transformed_block)?;
                if cost_after >= cost_before {
                    return Err(PaidBasisError(
                        "accepted exact-local basis move did not lower physical cost".into(),
                    ));
                }
                accepted_moves.push(AcceptedBasisMove {
                    sweep,
                    side,
                    group_indices,
                    cost_before,
                    cost_after,
                    preparation_range: preparation_start..preparation.len(),
                });
            }
        }
        if paid_block_cost(&preparation, &transformed_block)? == sweep_cost_before {
            break;
        }
    }

    Ok(OptimizationResult {
        transformed_block,
        source_basis,
        target_basis,
        preparation,
        source_preparation,
        target_preparation,
        tested_groups,
        accepted_moves,
    })
}
