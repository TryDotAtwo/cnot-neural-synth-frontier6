//! Deterministic local paid-basis shear compilation and endpoint auditing.

mod endpoint;
mod groups;
mod matrix;
mod word;

use super::{
    optimize_exact_basis, AcceptedBasisMove, BasisSide, PaidBasisError, PaidShearCandidate,
    PaidShearConfig,
};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
pub(crate) use endpoint::endpoint_sha256;
use endpoint::expected_shear_endpoint;
use groups::{local_vectors, select_groups, side_vectors};
use matrix::{identity_matrix, paid_block_cost, physical_wire, xor_rows};
pub(crate) use word::{cancel_adjacent_block_gates, ReducedWordAudit};

/// Compiles a binary shear block using deterministic, exact local bases of
/// width at most four. Each accepted move strictly decreases the paid physical
/// count. The resulting certificate covers the complete operator on `dimension`
/// physical wires; this routine makes no global-optimality claim.
pub fn compile_paid_shear(
    block: &[Vec<u8>],
    dimension: usize,
    source_wires: &[usize],
    target_wires: &[usize],
    config: PaidShearConfig,
) -> Result<PaidShearCandidate, PaidBasisError> {
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
    let mut physical_wires =
        std::collections::HashSet::with_capacity(source_wires.len() + target_wires.len());
    for &wire in source_wires.iter().chain(target_wires) {
        if wire >= dimension || !physical_wires.insert(wire) {
            return Err(PaidBasisError(
                "paid shear wires must be unique, disjoint, and inside the physical dimension"
                    .into(),
            ));
        }
    }
    let dimension_u32 = u32::try_from(dimension)
        .map_err(|_| PaidBasisError("physical dimension exceeds CNOT index range".into()))?;

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

    let alphabet = CnotAlphabet::all_to_all_unit(dimension_u32)
        .map_err(|error| PaidBasisError(error.to_string()))?;
    let start = MatrixState::identity(dimension_u32, alphabet.id())
        .map_err(|error| PaidBasisError(error.to_string()))?;
    let mut endpoint = start.clone();
    for gate in &gates {
        endpoint
            .apply_cnot(*gate)
            .map_err(|error| PaidBasisError(error.to_string()))?;
    }
    let expected =
        expected_shear_endpoint(&start, dimension_u32, source_wires, target_wires, block)?;
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
