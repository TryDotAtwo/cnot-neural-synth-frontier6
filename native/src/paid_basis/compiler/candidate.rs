//! Physical-word assembly and full-endpoint certificate construction.

use super::super::{PaidBasisError, PaidShearCandidate};
use super::endpoint::{endpoint_sha256, expected_shear_endpoint};
use super::matrix::physical_wire;
use super::optimize::OptimizationResult;
use super::word::{cancel_adjacent_block_gates, ReducedWordAudit};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};

pub(super) fn build_paid_shear_candidate(
    block: &[Vec<u8>],
    dimension: u32,
    source_wires: &[usize],
    target_wires: &[usize],
    optimization: OptimizationResult,
) -> Result<PaidShearCandidate, PaidBasisError> {
    let OptimizationResult {
        transformed_block,
        source_basis,
        target_basis,
        preparation,
        source_preparation,
        target_preparation,
        tested_groups,
        accepted_moves,
    } = optimization;

    let direct_cost = block.iter().flatten().map(|bit| usize::from(*bit)).sum();
    let mut core = Vec::new();
    for (target_index, row) in transformed_block.iter().enumerate() {
        for (source_index, bit) in row.iter().copied().enumerate() {
            if bit != 0 {
                core.push(CnotGate {
                    control: physical_wire(source_wires[source_index])?,
                    target: physical_wire(target_wires[target_index])?,
                });
            }
        }
    }
    let restoration = preparation.iter().rev().copied().collect::<Vec<_>>();
    let mut word_blocks = Vec::with_capacity(accepted_moves.len() * 2 + 1);
    for movement in &accepted_moves {
        word_blocks.push(preparation[movement.preparation_range.clone()].to_vec());
    }
    word_blocks.push(core.clone());
    for movement in accepted_moves.iter().rev() {
        word_blocks.push(
            preparation[movement.preparation_range.clone()]
                .iter()
                .rev()
                .copied()
                .collect(),
        );
    }
    let reduced: ReducedWordAudit = cancel_adjacent_block_gates(&word_blocks);
    let gates = reduced.word;

    let alphabet = CnotAlphabet::all_to_all_unit(dimension)
        .map_err(|error| PaidBasisError(error.to_string()))?;
    let start = MatrixState::identity(dimension, alphabet.id())
        .map_err(|error| PaidBasisError(error.to_string()))?;
    let mut endpoint = start.clone();
    for gate in &gates {
        endpoint
            .apply_cnot(*gate)
            .map_err(|error| PaidBasisError(error.to_string()))?;
    }
    let expected = expected_shear_endpoint(&start, dimension, source_wires, target_wires, block)?;
    if endpoint != expected {
        return Err(PaidBasisError(
            "compiled physical word does not realize the full declared shear endpoint".into(),
        ));
    }
    let paid_cost = gates.len();
    if paid_cost > direct_cost {
        return Err(PaidBasisError(
            "paid-basis candidate is worse than direct block realization".into(),
        ));
    }
    let certificate = WordCertificate {
        start,
        end: endpoint,
        physical_cost: u64::try_from(paid_cost)
            .map_err(|_| PaidBasisError("paid shear word length exceeds u64".into()))?,
        gates,
        producer: "own_shared_physical_basis_exact_local_v1".into(),
        lineage_id: "paid_shear_block".into(),
    };
    certificate
        .verify(&alphabet)
        .map_err(|error| PaidBasisError(error.to_string()))?;
    let endpoint_sha256 = endpoint_sha256(&certificate.end);

    Ok(PaidShearCandidate {
        certificate,
        preparation,
        source_preparation,
        target_preparation,
        core,
        restoration,
        original_block: block.to_vec(),
        transformed_block,
        source_basis,
        target_basis,
        direct_cost,
        paid_cost,
        boundary_cancellations: reduced.boundary_cancellations,
        endpoint_sha256,
        tested_groups,
        accepted_moves,
    })
}
