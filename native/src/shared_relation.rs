//! Exact I06 port of the pinned stage-56 labelled shared-relation feature.
//!
//! The provider retains the full relation kernel of selected physical rows and
//! each labelled readout through an explicit independent covector frame. It is
//! a boundary signature, not a generic network contraction or CNOT-distance
//! estimate.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

/// SHA-256 of the exact stage-56 source used by the Python implementation.
pub const STAGE56_SOURCE_SHA256: &str =
    "b14e85497632860665754a4014c34a4bcf149a9f29a1af6cd3710f401559c7ef";

const DEFAULT_MAX_VECTORS: usize = 32;
const DEFAULT_MAX_COORDINATES: usize = 144;
const DEFAULT_MAX_PORTS: usize = 32;
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const EXACT_F32_INTEGER_LIMIT: usize = 1 << 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRelationContextError(pub String);

impl fmt::Display for SharedRelationContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SharedRelationContextError {}

/// Explicit physical row vectors and a fixed labelled covector frame.
///
/// `covectors[j]` is packed little-endian in the order of `coordinates`:
/// coordinate 0 is bit 0 of word 0. `rows` and `coordinates` preserve caller
/// order; that order is part of the I06 feature identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRelationContext {
    n: u32,
    rows: Vec<u32>,
    coordinates: Vec<u32>,
    covectors: Vec<Vec<u64>>,
}

impl SharedRelationContext {
    pub fn new(
        n: u32,
        rows: Vec<u32>,
        coordinates: Vec<u32>,
        covectors: Vec<Vec<u64>>,
    ) -> Result<Self, SharedRelationContextError> {
        if n == 0 {
            return Err(SharedRelationContextError(
                "positive physical width required".into(),
            ));
        }
        validate_indices(&rows, n, "selected rows")?;
        validate_indices(&coordinates, n, "ambient coordinates")?;
        let coordinate_words = words_for(coordinates.len())
            .ok_or_else(|| SharedRelationContextError("coordinate width overflow".into()))?;
        if covectors.iter().any(|vector| {
            vector.len() != coordinate_words || has_bits_above(vector, coordinates.len())
        }) {
            return Err(SharedRelationContextError(
                "covectors must fit the declared coordinate width".into(),
            ));
        }
        if canonical_basis(&covectors, coordinates.len()).len() != covectors.len() {
            return Err(SharedRelationContextError(
                "covector frame must be linearly independent".into(),
            ));
        }
        Ok(Self {
            n,
            rows,
            coordinates,
            covectors,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    pub fn coordinates(&self) -> &[u32] {
        &self.coordinates
    }

    pub fn covectors(&self) -> &[Vec<u64>] {
        &self.covectors
    }
}

fn validate_indices(
    indices: &[u32],
    n: u32,
    label: &str,
) -> Result<(), SharedRelationContextError> {
    let mut seen = HashSet::with_capacity(indices.len());
    if indices.is_empty()
        || indices
            .iter()
            .any(|index| *index >= n || !seen.insert(*index))
    {
        return Err(SharedRelationContextError(format!(
            "{label} must be nonempty, distinct, and within physical width"
        )));
    }
    Ok(())
}

fn words_for(width: usize) -> Option<usize> {
    width.checked_add(63).map(|value| value / 64)
}

fn bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

fn set_bit(words: &mut [u64], index: usize) {
    words[index / 64] |= 1u64 << (index % 64);
}

fn highest_bit(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().rev().find_map(|(index, word)| {
        (*word != 0).then(|| index * 64 + (63 - word.leading_zeros() as usize))
    })
}

fn has_bits_above(words: &[u64], width: usize) -> bool {
    if width == 0 {
        return words.iter().any(|word| *word != 0);
    }
    let full_words = width / 64;
    if words
        .iter()
        .skip(full_words + usize::from(width % 64 != 0))
        .any(|v| *v != 0)
    {
        return true;
    }
    let tail = width % 64;
    tail != 0
        && words
            .get(full_words)
            .is_some_and(|word| word & !((1u64 << tail) - 1) != 0)
}

/// Canonical reduced basis with highest-set-bit pivots, matching stage-56.
fn canonical_basis(vectors: &[Vec<u64>], width: usize) -> Vec<Vec<u64>> {
    let word_count = words_for(width).unwrap_or_default();
    let mut pivots = BTreeMap::<usize, Vec<u64>>::new();
    for raw in vectors {
        let mut row = vec![0u64; word_count];
        for (target, source) in row.iter_mut().zip(raw) {
            *target = *source;
        }
        for (pivot, basis_row) in pivots.iter().rev() {
            if bit(&row, *pivot) {
                xor_assign(&mut row, basis_row);
            }
        }
        let Some(pivot) = highest_bit(&row) else {
            continue;
        };
        for basis_row in pivots.values_mut() {
            if bit(basis_row, pivot) {
                xor_assign(basis_row, &row);
            }
        }
        pivots.insert(pivot, row);
    }
    pivots.into_iter().rev().map(|(_, row)| row).collect()
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= *b;
    }
}

fn nullspace(equations: &[Vec<u64>], width: usize) -> Vec<Vec<u64>> {
    let reduced = canonical_basis(equations, width);
    let pivot_set: HashSet<usize> = reduced.iter().filter_map(|row| highest_bit(row)).collect();
    let mut basis = Vec::new();
    for free in 0..width {
        if pivot_set.contains(&free) {
            continue;
        }
        let mut vector = vec![0u64; words_for(width).unwrap_or_default()];
        set_bit(&mut vector, free);
        for row in &reduced {
            let pivot = highest_bit(row).expect("reduced basis rows are nonzero");
            if bit(row, free) {
                set_bit(&mut vector, pivot);
            }
        }
        basis.push(vector);
    }
    canonical_basis(&basis, width)
}

struct SharedRelationProducer {
    spec: FeatureSpec,
    contexts: Vec<SharedRelationContext>,
    max_vectors: usize,
    max_coordinates: usize,
    max_ports: usize,
}

impl FeatureProducer for SharedRelationProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "i06_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i06_context_for_matrix_width",
            );
        };
        let (n, vectors, coordinates, ports) = (
            usize::try_from(context.n).unwrap_or(usize::MAX),
            context.rows.len(),
            context.coordinates.len(),
            context.covectors.len(),
        );
        if vectors > self.max_vectors
            || coordinates > self.max_coordinates
            || ports > self.max_ports
        {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i06_context_exceeds_declared_vector_coordinate_or_port_budget",
            );
        }
        let Some(estimate) =
            working_bytes_upper_bound(n, vectors, coordinates, ports, self.max_vectors)
        else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i06_working_memory_size_overflow",
            );
        };
        if estimate > policy.max_bytes.unwrap_or(DEFAULT_MAX_BYTES) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i06_working_memory_budget_exceeded_before_allocation",
            );
        }

        match compute_tokens(state, context, self.max_vectors, estimate) {
            Ok(value) => FeatureComputation::Computed(value),
            Err(reason) => unavailable(FeatureStatus::ResourceLimit, reason),
        }
    }
}

fn working_bytes_upper_bound(
    n: usize,
    vectors: usize,
    coordinates: usize,
    ports: usize,
    max_vectors: usize,
) -> Option<usize> {
    let square = n.checked_mul(n)?;
    let context_work = coordinates
        .checked_mul(vectors)?
        .checked_add(vectors.checked_mul(vectors)?)?
        .checked_mul(16)?;
    let group_work = vectors
        .checked_add(ports)?
        .checked_mul(max_vectors.checked_add(2)?.checked_add(n.checked_mul(2)?)?)?
        .checked_mul(4)?;
    square
        .checked_mul(12)?
        .checked_add(context_work)?
        .checked_add(group_work)
}

fn compute_tokens(
    state: &MatrixState,
    context: &SharedRelationContext,
    max_vectors: usize,
    estimated_bytes: usize,
) -> Result<FeatureValue, String> {
    let vector_count = context.rows.len();
    let coordinate_count = context.coordinates.len();
    let coordinate_words = words_for(coordinate_count).ok_or("i06_coordinate_width_overflow")?;
    let row_words = words_for(vector_count).ok_or("i06_row_width_overflow")?;
    let mut vectors = Vec::new();
    vectors
        .try_reserve_exact(vector_count)
        .map_err(|_| "i06_vector_index_allocation_failed".to_string())?;
    for physical_row in &context.rows {
        let row = state
            .row_words(*physical_row)
            .ok_or_else(|| "i06_selected_physical_row_is_out_of_range".to_string())?;
        let mut vector = vec![0u64; coordinate_words];
        for (coordinate_index, physical_coordinate) in context.coordinates.iter().enumerate() {
            if bit(row, *physical_coordinate as usize) {
                set_bit(&mut vector, coordinate_index);
            }
        }
        vectors.push(vector);
    }

    let mut equations = Vec::new();
    equations
        .try_reserve_exact(coordinate_count)
        .map_err(|_| "i06_equation_index_allocation_failed".to_string())?;
    for coordinate in 0..coordinate_count {
        let mut equation = vec![0u64; row_words];
        for (index, vector) in vectors.iter().enumerate() {
            if bit(vector, coordinate) {
                set_bit(&mut equation, index);
            }
        }
        equations.push(equation);
    }
    let kernel = nullspace(&equations, vector_count);

    let mut port_masks = Vec::new();
    port_masks
        .try_reserve_exact(context.covectors.len())
        .map_err(|_| "i06_port_index_allocation_failed".to_string())?;
    for covector in &context.covectors {
        let mut mask = vec![0u64; row_words];
        for (index, vector) in vectors.iter().enumerate() {
            let parity = covector
                .iter()
                .zip(vector)
                .fold(0u32, |acc, (a, b)| acc ^ ((a & b).count_ones() & 1));
            if parity != 0 {
                set_bit(&mut mask, index);
            }
        }
        port_masks.push(mask);
    }

    let group_count = kernel
        .len()
        .checked_add(port_masks.len())
        .ok_or("i06_group_count_overflow")?;
    let channels = max_vectors
        .checked_add(2)
        .ok_or("i06_channel_count_overflow")?;
    let value_count = group_count
        .checked_mul(channels)
        .ok_or("i06_feature_value_count_overflow")?;
    let n = usize::try_from(context.n).map_err(|_| "i06_physical_width_overflow")?;
    let incidence_count = group_count
        .checked_mul(n)
        .and_then(|value| value.checked_mul(2))
        .ok_or("i06_incidence_count_overflow")?;
    let mut values = allocate_zeroed::<f32>(value_count, "i06_feature_values_allocation_failed")?;
    let mut token_valid = allocate_zeroed::<u8>(group_count, "i06_token_mask_allocation_failed")?;
    let mut incidence = allocate_zeroed::<f32>(incidence_count, "i06_incidence_allocation_failed")?;

    for (group, relation) in kernel.iter().enumerate() {
        let base = group * channels;
        values[base] = 0.0;
        values[base + 1] = group as f32;
        for index in 0..vector_count {
            values[base + 2 + index] = f32::from(bit(relation, index));
            if bit(relation, index) {
                set_incidence(&mut incidence, group, n, context.rows[index] as usize, 0);
            }
        }
        token_valid[group] = 1;
    }
    for (label, relation) in port_masks.iter().enumerate() {
        let group = kernel.len() + label;
        let base = group * channels;
        values[base] = 1.0;
        values[base + 1] = label as f32;
        for index in 0..vector_count {
            values[base + 2 + index] = f32::from(bit(relation, index));
            if bit(relation, index) {
                set_incidence(&mut incidence, group, n, context.rows[index] as usize, 0);
            }
        }
        for (coordinate_index, physical_coordinate) in context.coordinates.iter().enumerate() {
            if bit(&context.covectors[label], coordinate_index) {
                set_incidence(&mut incidence, group, n, *physical_coordinate as usize, 1);
            }
        }
        token_valid[group] = 1;
    }

    debug_assert!(estimated_bytes >= value_count * 4 + incidence_count * 4);
    Ok(FeatureValue {
        shape: vec![group_count, channels],
        values,
        token_valid,
        incidence,
    })
}

fn allocate_zeroed<T: Clone + Default>(count: usize, reason: &str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| reason.to_string())?;
    values.resize(count, T::default());
    Ok(values)
}

fn set_incidence(incidence: &mut [f32], group: usize, nodes: usize, node: usize, role: usize) {
    incidence[(group * nodes + node) * 2 + role] = 1.0;
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

/// Registers I06 for explicit, labelled physical contexts.
pub fn register_shared_relation_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<SharedRelationContext>,
    max_vectors: usize,
    max_coordinates: usize,
    max_ports: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty()
        || max_vectors == 0
        || max_coordinates == 0
        || max_ports == 0
        || max_vectors > EXACT_F32_INTEGER_LIMIT
        || max_ports > EXACT_F32_INTEGER_LIMIT
        || max_vectors.checked_add(2).is_none()
    {
        return Err(FeatureBuildError(
            "I06 requires contexts and positive finite budgets with exact float labels".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I06 allows one labelled context per physical width".into(),
        ));
    }
    let producer_version = producer_version(&contexts, max_vectors, max_coordinates, max_ports);
    let id = FeatureId::new(b'I', 6).expect("I06 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = max_vectors + 2;
    let spec = spec.with_provider_metadata(
        "selected_physical_row_vectors;full_labelled_kernel_K_and_UX_bits;fixed_independent_U;no_rank_only_replacement",
        producer_version,
        ["M", "labelled_rows", "ambient_coordinates", "covector_frame"],
        "exact",
        "T2",
        ["vector_row", "covector_coordinate"],
        false,
    );
    registry.register(Box::new(SharedRelationProducer {
        spec,
        contexts,
        max_vectors,
        max_coordinates,
        max_ports,
    }))
}

pub fn register_default_shared_relation_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<SharedRelationContext>,
) -> Result<(), FeatureBuildError> {
    register_shared_relation_contexts(
        registry,
        contexts,
        DEFAULT_MAX_VECTORS,
        DEFAULT_MAX_COORDINATES,
        DEFAULT_MAX_PORTS,
    )
}

fn producer_version(
    contexts: &[SharedRelationContext],
    max_vectors: usize,
    max_coordinates: usize,
    max_ports: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"stage56_full_shared_relation_v1");
    hash_field(&mut digest, STAGE56_SOURCE_SHA256.as_bytes());
    hash_field(&mut digest, &max_vectors.to_le_bytes());
    hash_field(&mut digest, &max_coordinates.to_le_bytes());
    hash_field(&mut digest, &max_ports.to_le_bytes());
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_indices(&mut digest, &context.rows);
        hash_indices(&mut digest, &context.coordinates);
        hash_field(&mut digest, &(context.covectors.len() as u64).to_le_bytes());
        for covector in &context.covectors {
            hash_field(&mut digest, &(covector.len() as u64).to_le_bytes());
            for word in covector {
                hash_field(&mut digest, &word.to_le_bytes());
            }
        }
    }
    format!(
        "stage56_full_shared_relation_v1:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    hash_field(digest, &(indices.len() as u64).to_le_bytes());
    for index in indices {
        hash_field(digest, &index.to_le_bytes());
    }
}
