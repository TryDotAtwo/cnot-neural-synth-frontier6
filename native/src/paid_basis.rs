//! Exact small-width paid-basis optimization from the source local atlas.
//!
//! The atlas covers every invertible binary coordinate change on 1..=4 local
//! wires. Its price includes preparation and its physical inverse/restoration,
//! both counted in CNOTs. It optimizes a block's local basis only; it is not a
//! global CNOT shortest-path solver.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ops::Range;
use std::sync::OnceLock;

use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::{CnotGate, MatrixState};
use sha2::{Digest, Sha256};

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

#[derive(Debug)]
struct ExactAtlas {
    distance: Vec<u8>,
    parent: Vec<Option<usize>>,
    parent_gate: Vec<Option<(usize, usize)>>,
    population: Vec<u8>,
}

struct AtlasCache {
    source: [OnceLock<ExactAtlas>; 5],
    target: [OnceLock<ExactAtlas>; 5],
}

static ATLAS_CACHE: OnceLock<AtlasCache> = OnceLock::new();

/// Returns the exact minimum paid price over every local GL(width, 2) basis.
///
/// `vectors` stores one binary vector per row. For `Source`, a gate `(c,t)`
/// changes each row by `row[c] ^= row[t]`; for `Target` it changes
/// `row[t] ^= row[c]`. The gate order and equal-cost tie break match the
/// source `code/exact_basis.py` atlas construction.
pub fn optimize_exact_basis(
    vectors: &[Vec<u8>],
    side: BasisSide,
) -> Result<ExactBasisOptimization, PaidBasisError> {
    let width = vectors
        .first()
        .map(Vec::len)
        .ok_or_else(|| PaidBasisError("basis block must contain at least one vector".into()))?;
    if !(1..=4).contains(&width) {
        return Err(PaidBasisError(
            "exact paid-basis atlas supports local width 1..=4".into(),
        ));
    }
    if vectors
        .iter()
        .any(|row| row.len() != width || row.iter().any(|bit| *bit > 1))
    {
        return Err(PaidBasisError(
            "basis vectors must have a consistent width and contain only bits".into(),
        ));
    }

    let mut direct_cost = 0usize;
    let mut histogram = vec![0usize; 1usize << width];
    for row in vectors {
        let mut code = 0usize;
        for (bit_index, bit) in row.iter().copied().enumerate() {
            direct_cost = direct_cost
                .checked_add(usize::from(bit))
                .ok_or_else(|| PaidBasisError("direct basis cost overflow".into()))?;
            code |= usize::from(bit) << bit_index;
        }
        histogram[code] = histogram[code]
            .checked_add(1)
            .ok_or_else(|| PaidBasisError("basis histogram count overflow".into()))?;
    }

    let atlas = atlas(width, side);
    let mut best_index = 0usize;
    let mut best_cost = usize::MAX;
    let code_count = histogram.len();
    for (state_index, population) in atlas.population.chunks_exact(code_count).enumerate() {
        let mut transformed_core_cost = 0usize;
        for (count, transformed_weight) in histogram.iter().copied().zip(population.iter().copied())
        {
            if count == 0 {
                continue;
            }
            let term = usize::from(transformed_weight)
                .checked_mul(count)
                .ok_or_else(|| PaidBasisError("transformed core cost overflow".into()))?;
            transformed_core_cost = transformed_core_cost
                .checked_add(term)
                .ok_or_else(|| PaidBasisError("transformed core cost overflow".into()))?;
        }
        let preparation_cost = usize::from(atlas.distance[state_index])
            .checked_mul(2)
            .ok_or_else(|| PaidBasisError("paid preparation cost overflow".into()))?;
        let paid_cost = preparation_cost
            .checked_add(transformed_core_cost)
            .ok_or_else(|| PaidBasisError("paid basis cost overflow".into()))?;
        // The source uses first-occurrence argmin. Iterating atlas BFS order and
        // replacing only on strict improvement preserves that tie break.
        if paid_cost < best_cost {
            best_cost = paid_cost;
            best_index = state_index;
        }
    }

    let mut operations = Vec::new();
    let mut cursor = best_index;
    while let Some(parent) = atlas.parent[cursor] {
        operations.push(atlas.parent_gate[cursor].ok_or_else(|| {
            PaidBasisError("exact atlas parent is missing its generating gate".into())
        })?);
        cursor = parent;
    }
    operations.reverse();

    let mut transformed_vectors = vectors.to_vec();
    for &(control, target) in &operations {
        for row in &mut transformed_vectors {
            match side {
                BasisSide::Source => row[control] ^= row[target],
                BasisSide::Target => row[target] ^= row[control],
            }
        }
    }
    let independently_counted = operations
        .len()
        .checked_mul(2)
        .and_then(|preparation| {
            transformed_vectors
                .iter()
                .flatten()
                .try_fold(preparation, |sum, bit| sum.checked_add(usize::from(*bit)))
        })
        .ok_or_else(|| PaidBasisError("verified paid basis cost overflow".into()))?;
    if independently_counted != best_cost || best_cost > direct_cost {
        return Err(PaidBasisError(
            "exact atlas candidate failed its paid-cost invariant".into(),
        ));
    }

    Ok(ExactBasisOptimization {
        operations,
        direct_cost,
        paid_cost: best_cost,
        transformed_vectors,
    })
}

fn atlas(width: usize, side: BasisSide) -> &'static ExactAtlas {
    let cache = ATLAS_CACHE.get_or_init(|| AtlasCache {
        source: std::array::from_fn(|_| OnceLock::new()),
        target: std::array::from_fn(|_| OnceLock::new()),
    });
    let slots = match side {
        BasisSide::Source => &cache.source,
        BasisSide::Target => &cache.target,
    };
    slots[width].get_or_init(|| ExactAtlas::build(width, side))
}

impl ExactAtlas {
    fn build(width: usize, side: BasisSide) -> Self {
        debug_assert!((1..=4).contains(&width));
        let state_count = 1usize << width;
        let identity = (0..state_count).collect::<Vec<_>>();
        let identity_key = encode_mapping(&identity, width);
        let mut states = vec![identity_key];
        let mut distance = vec![0u8];
        let mut parent = vec![None];
        let mut parent_gate = vec![None];
        let mut seen = HashMap::new();
        seen.insert(identity_key, 0usize);
        let mut queue = VecDeque::from([0usize]);

        while let Some(index) = queue.pop_front() {
            let state = states[index];
            for control in 0..width {
                for target in 0..width {
                    if control == target {
                        continue;
                    }
                    let next = apply_to_mapping(state, width, side, control, target);
                    if seen.contains_key(&next) {
                        continue;
                    }
                    let next_index = states.len();
                    seen.insert(next, next_index);
                    states.push(next);
                    distance.push(distance[index] + 1);
                    parent.push(Some(index));
                    parent_gate.push(Some((control, target)));
                    queue.push_back(next_index);
                }
            }
        }

        let mut population = Vec::with_capacity(states.len() * state_count);
        for state in &states {
            for code in 0..state_count {
                population.push(state_image(*state, width, code).count_ones() as u8);
            }
        }

        Self {
            distance,
            parent,
            parent_gate,
            population,
        }
    }
}

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
    let reduced = cancel_adjacent_block_gates(&word_blocks);
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

#[derive(Debug)]
pub(crate) struct ReducedWordAudit {
    pub(crate) word: Vec<CnotGate>,
    pub(crate) boundary_cancellations: Vec<BoundaryCancellation>,
}

pub(crate) fn cancel_adjacent_block_gates(blocks: &[Vec<CnotGate>]) -> ReducedWordAudit {
    let mut stack = Vec::<(CnotGate, usize)>::new();
    let mut boundary_counts = HashMap::<(u32, u32), usize>::new();
    for (block_index, block) in blocks.iter().enumerate() {
        for gate in block.iter().copied() {
            if stack.last().is_some_and(|(previous, _)| *previous == gate) {
                let (_, previous_block) = stack.pop().expect("the prior matching gate exists");
                if previous_block != block_index {
                    *boundary_counts
                        .entry((gate.control, gate.target))
                        .or_default() += 1;
                }
            } else {
                stack.push((gate, block_index));
            }
        }
    }
    let mut boundary_cancellations = boundary_counts
        .into_iter()
        .map(|((control, target), canceled_pairs)| BoundaryCancellation {
            control,
            target,
            canceled_pairs,
        })
        .collect::<Vec<_>>();
    boundary_cancellations.sort_by_key(|item| (item.control, item.target));
    ReducedWordAudit {
        word: stack.into_iter().map(|(gate, _)| gate).collect(),
        boundary_cancellations,
    }
}

pub(crate) fn endpoint_sha256(state: &MatrixState) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"cnot103_paid_shear_endpoint_v1");
    hash_field(&mut digest, &state.dimension().to_le_bytes());
    hash_field(&mut digest, state.alphabet_id().as_bytes());
    hash_field(
        &mut digest,
        &(state.packed_rows().len() as u64).to_le_bytes(),
    );
    for word in state.packed_rows() {
        hash_field(&mut digest, &word.to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_field(digest: &mut Sha256, field: &[u8]) {
    digest.update((field.len() as u64).to_le_bytes());
    digest.update(field);
}

fn physical_wire(wire: usize) -> Result<u32, PaidBasisError> {
    u32::try_from(wire).map_err(|_| PaidBasisError("wire index exceeds u32 range".into()))
}

fn identity_matrix(width: usize) -> Vec<Vec<u8>> {
    (0..width)
        .map(|row| (0..width).map(|column| u8::from(row == column)).collect())
        .collect()
}

fn xor_rows(matrix: &mut [Vec<u8>], target: usize, control: usize) {
    let source = matrix[control].clone();
    for (value, source) in matrix[target].iter_mut().zip(source) {
        *value ^= source;
    }
}

fn block_hamming_weight(block: &[Vec<u8>]) -> usize {
    block.iter().flatten().map(|bit| usize::from(*bit)).sum()
}

fn paid_block_cost(preparation: &[CnotGate], block: &[Vec<u8>]) -> Result<usize, PaidBasisError> {
    preparation
        .len()
        .checked_mul(2)
        .and_then(|prep| prep.checked_add(block_hamming_weight(block)))
        .ok_or_else(|| PaidBasisError("paid partial-block cost overflow".into()))
}

fn side_vectors(block: &[Vec<u8>], side: BasisSide) -> Vec<Vec<u8>> {
    match side {
        BasisSide::Source => (0..block[0].len())
            .map(|column| block.iter().map(|row| row[column]).collect())
            .collect(),
        BasisSide::Target => block.to_vec(),
    }
}

fn local_vectors(block: &[Vec<u8>], group_indices: &[usize], side: BasisSide) -> Vec<Vec<u8>> {
    match side {
        BasisSide::Source => block
            .iter()
            .map(|row| group_indices.iter().map(|&index| row[index]).collect())
            .collect(),
        BasisSide::Target => (0..block[0].len())
            .map(|column| {
                group_indices
                    .iter()
                    .map(|&row| block[row][column])
                    .collect()
            })
            .collect(),
    }
}

fn select_groups(vectors: &[Vec<u8>], max_width: usize) -> Vec<Vec<usize>> {
    if vectors.len() < 2 {
        return Vec::new();
    }
    let weights = vectors
        .iter()
        .map(|row| row.iter().map(|bit| usize::from(*bit)).sum::<usize>())
        .collect::<Vec<_>>();
    let active = weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| (*weight > 0).then_some(index))
        .collect::<Vec<_>>();
    if active.len() < 2 {
        return Vec::new();
    }
    let width = max_width.min(active.len());
    let mut leaders = active.clone();
    leaders.sort_by_key(|index| (std::cmp::Reverse(weights[*index]), *index));
    let mut seen = std::collections::HashSet::<Vec<usize>>::new();
    let mut groups = Vec::new();

    for leader in leaders {
        let mut others = active
            .iter()
            .copied()
            .filter(|index| *index != leader)
            .collect::<Vec<_>>();
        others.sort_by_key(|other| {
            let distance = vectors[leader]
                .iter()
                .zip(&vectors[*other])
                .filter(|(left, right)| left != right)
                .count() as isize;
            (
                distance - weights[leader].max(weights[*other]) as isize,
                *other,
            )
        });
        let mut group = Vec::with_capacity(width);
        group.push(leader);
        group.extend(others.into_iter().take(width - 1));
        group.sort_unstable();
        if seen.insert(group.clone()) {
            groups.push(group);
        }
    }

    let step = width.saturating_sub(1).max(1);
    for start in (0..active.len()).step_by(step) {
        let end = (start + width).min(active.len());
        let group = active[start..end].to_vec();
        if group.len() > 1 && seen.insert(group.clone()) {
            groups.push(group);
        }
    }
    groups
}

fn expected_shear_endpoint(
    start: &MatrixState,
    dimension: u32,
    source_wires: &[usize],
    target_wires: &[usize],
    block: &[Vec<u8>],
) -> Result<MatrixState, PaidBasisError> {
    let words_per_row = start.words_per_row();
    let row_count = usize::try_from(dimension)
        .map_err(|_| PaidBasisError("dimension cannot be represented as usize".into()))?;
    let packed_len = row_count
        .checked_mul(words_per_row)
        .ok_or_else(|| PaidBasisError("expected endpoint allocation overflow".into()))?;
    let mut rows = vec![0u64; packed_len];
    for wire in 0..row_count {
        rows[wire * words_per_row + wire / 64] |= 1u64 << (wire % 64);
    }
    for (target_index, target_wire) in target_wires.iter().copied().enumerate() {
        for (source_index, source_wire) in source_wires.iter().copied().enumerate() {
            if block[target_index][source_index] != 0 {
                let offset = target_wire * words_per_row + source_wire / 64;
                rows[offset] ^= 1u64 << (source_wire % 64);
            }
        }
    }
    MatrixState::from_packed_rows(dimension, start.alphabet_id(), rows)
        .map_err(|error| PaidBasisError(error.to_string()))
}

fn encode_mapping(mapping: &[usize], width: usize) -> u64 {
    mapping
        .iter()
        .enumerate()
        .fold(0u64, |packed, (input, output)| {
            packed | ((*output as u64) << (input * width))
        })
}

fn state_image(state: u64, width: usize, code: usize) -> usize {
    let mask = (1u64 << width) - 1;
    ((state >> (code * width)) & mask) as usize
}

fn apply_to_mapping(
    state: u64,
    width: usize,
    side: BasisSide,
    control: usize,
    target: usize,
) -> u64 {
    let mut packed = 0u64;
    for input in 0..(1usize << width) {
        let mut output = state_image(state, width, input);
        match side {
            BasisSide::Source => output ^= ((output >> target) & 1) << control,
            BasisSide::Target => output ^= ((output >> control) & 1) << target,
        }
        packed |= (output as u64) << (input * width);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::{atlas, cancel_adjacent_block_gates, BasisSide};
    use crate::state::CnotGate;

    #[test]
    fn atlas_enumerates_every_local_linear_basis() {
        let expected = [0usize, 1, 6, 168, 20_160];
        for width in 1..=4 {
            assert_eq!(
                atlas(width, BasisSide::Source).population.len() / (1usize << width),
                expected[width]
            );
            assert_eq!(
                atlas(width, BasisSide::Target).population.len() / (1usize << width),
                expected[width]
            );
        }
    }

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
