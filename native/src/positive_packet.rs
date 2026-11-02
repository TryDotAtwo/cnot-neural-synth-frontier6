//! Exact I08 port of the pinned stage-59 affine optional-CNOT packet.
//!
//! The feature describes cut widths and still-live selector labels. It is not
//! a distance oracle and does not silently linearize nonlinear endpoint maps.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::mem::size_of;

/// SHA-256 of the stage-59 source used by the research-v3 I08 implementation.
pub const STAGE59_SOURCE_SHA256: &str =
    "63824b056fa1983a10b3d535532c2975c7c3a1c96cfb2a90267ead2a1b168987";

const DEFAULT_MAX_N: usize = 144;
const DEFAULT_MAX_SLOTS: usize = 256;
const DEFAULT_MAX_MONOMIALS: usize = 4096;
const DEFAULT_MAX_FACTOR_ENTRIES: usize = 65_536;
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositivePacketContextError(pub String);

impl fmt::Display for PositivePacketContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PositivePacketContextError {}

/// One Boolean tensor factor over ordered selector identities.
///
/// Bit `j` of each allowed `BigUint` labels the assignment of `scope[j]`.
/// Scope order is significant and is preserved in the producer identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalBooleanFactor {
    scope: Vec<u32>,
    allowed: Vec<BigUint>,
}

impl LocalBooleanFactor {
    pub fn new(scope: Vec<u32>, allowed: Vec<BigUint>) -> Result<Self, PositivePacketContextError> {
        let mut seen = HashSet::with_capacity(scope.len());
        if scope.iter().any(|index| !seen.insert(*index)) {
            return Err(PositivePacketContextError(
                "factor selector scope must contain distinct indices".into(),
            ));
        }
        if allowed.windows(2).any(|pair| pair[0] >= pair[1])
            || allowed
                .iter()
                .any(|pattern| pattern.bits() > scope.len() as u64)
        {
            return Err(PositivePacketContextError(
                "allowed factor patterns must be sorted, unique, and fit the scope".into(),
            ));
        }
        Ok(Self { scope, allowed })
    }

    pub fn scope(&self) -> &[u32] {
        &self.scope
    }

    pub fn allowed(&self) -> &[BigUint] {
        &self.allowed
    }
}

/// Fixed physical order of optional CNOT selectors and their local factors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionalGatePacket {
    n: u32,
    gates: Vec<(u32, u32)>,
    factors: Vec<LocalBooleanFactor>,
}

impl OptionalGatePacket {
    pub fn new(
        n: u32,
        gates: Vec<(u32, u32)>,
        factors: Vec<LocalBooleanFactor>,
    ) -> Result<Self, PositivePacketContextError> {
        if n == 0 {
            return Err(PositivePacketContextError(
                "positive physical width required".into(),
            ));
        }
        if gates
            .iter()
            .any(|(control, target)| *control >= n || *target >= n || control == target)
        {
            return Err(PositivePacketContextError(
                "each selector must name a valid physical CNOT".into(),
            ));
        }
        let slots = u32::try_from(gates.len()).map_err(|_| {
            PositivePacketContextError("selector count exceeds the index representation".into())
        })?;
        if factors
            .iter()
            .flat_map(|factor| &factor.scope)
            .any(|selector| *selector >= slots)
        {
            return Err(PositivePacketContextError(
                "factor selector index is outside the ordered gate packet".into(),
            ));
        }
        Ok(Self { n, gates, factors })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn gates(&self) -> &[(u32, u32)] {
        &self.gates
    }

    pub fn factors(&self) -> &[LocalBooleanFactor] {
        &self.factors
    }
}

struct PositivePacketProducer {
    spec: FeatureSpec,
    contexts: Vec<OptionalGatePacket>,
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
}

impl FeatureProducer for PositivePacketProducer {
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
                "i08_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i08_context_for_matrix_width",
            );
        };
        let n = usize::try_from(context.n).unwrap_or(usize::MAX);
        let slots = context.gates.len();
        if n > self.max_n
            || slots > self.max_slots
            || factor_entry_count(&context.factors)
                .is_none_or(|count| count > self.max_factor_entries)
        {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i08_context_exceeds_declared_width_slot_or_factor_budget",
            );
        }
        let expected_alphabet = format!("all_to_all_unit_cnot:{n}");
        if state.alphabet_id() != expected_alphabet {
            return unavailable(
                FeatureStatus::Unsupported,
                "i08_requires_matching_all_to_all_unit_cnot_alphabet",
            );
        }
        let Some(estimate) = working_bytes_upper_bound(n, slots, self.max_monomials) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i08_working_memory_size_overflow",
            );
        };
        if estimate > policy.max_bytes.unwrap_or(DEFAULT_MAX_BYTES) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i08_working_memory_budget_exceeded_before_allocation",
            );
        }

        match compute_values(state, context, self.max_monomials) {
            Ok(value) => FeatureComputation::Computed(value),
            Err(ComputeFailure::ResourceLimit) => unavailable(
                FeatureStatus::ResourceLimit,
                "i08_endpoint_polynomial_or_allocation_exceeded_declared_budget",
            ),
            Err(ComputeFailure::NonAffine) => {
                unavailable(FeatureStatus::Unsupported, "i08_endpoint_map_is_non_affine")
            }
            Err(ComputeFailure::Internal) => unavailable(
                FeatureStatus::Unsupported,
                "i08_internal_rank_or_shape_invariant_failed",
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComputeFailure {
    ResourceLimit,
    NonAffine,
    Internal,
}

#[derive(Clone, Debug)]
struct PolynomialTerm {
    mask: Vec<u64>,
    matrix: Vec<u64>,
}

fn compute_values(
    state: &MatrixState,
    context: &OptionalGatePacket,
    max_monomials: usize,
) -> Result<FeatureValue, ComputeFailure> {
    let n = usize::try_from(context.n).map_err(|_| ComputeFailure::ResourceLimit)?;
    let slots = context.gates.len();
    let words_per_row = n.checked_add(63).ok_or(ComputeFailure::ResourceLimit)? / 64;
    let matrix_words = n
        .checked_mul(words_per_row)
        .ok_or(ComputeFailure::ResourceLimit)?;
    let mask_words = words_for(slots).ok_or(ComputeFailure::ResourceLimit)?;

    // Expand (I + x_j E_j) in the exact declared physical gate order. A term
    // produced at slot j receives a fresh bit, so distinct histories cannot
    // collide in the polynomial map.
    let mut identity = zeroed_u64(matrix_words)?;
    for row in 0..n {
        identity[row * words_per_row + row / 64] |= 1u64 << (row % 64);
    }
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(1)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    terms.push(PolynomialTerm {
        mask: zeroed_u64(mask_words)?,
        matrix: identity,
    });

    for (slot, (control, target)) in context.gates.iter().copied().enumerate() {
        let control = usize::try_from(control).map_err(|_| ComputeFailure::ResourceLimit)?;
        let target = usize::try_from(target).map_err(|_| ComputeFailure::ResourceLimit)?;
        let original_len = terms.len();
        let remaining = max_monomials.saturating_sub(original_len);
        let mut extra = Vec::new();
        extra
            .try_reserve_exact(remaining.min(original_len))
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        for term in &terms[..original_len] {
            let source_start = control
                .checked_mul(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            let source_end = source_start
                .checked_add(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            let source_row = &term.matrix[source_start..source_end];
            if source_row.iter().all(|word| *word == 0) {
                continue;
            }
            if extra.len() >= remaining {
                return Err(ComputeFailure::ResourceLimit);
            }
            let mut mask = term.mask.clone();
            set_bit(&mut mask, slot);
            let mut matrix = zeroed_u64(matrix_words)?;
            let target_start = target
                .checked_mul(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            let target_end = target_start
                .checked_add(words_per_row)
                .ok_or(ComputeFailure::ResourceLimit)?;
            matrix[target_start..target_end].copy_from_slice(source_row);
            extra.push(PolynomialTerm { mask, matrix });
        }
        terms
            .try_reserve(extra.len())
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        terms.extend(extra);
    }

    let flat_bits = n.checked_mul(n).ok_or(ComputeFailure::ResourceLimit)?;
    let flat_words = words_for(flat_bits).ok_or(ComputeFailure::ResourceLimit)?;
    let mut singleton_terms = Vec::new();
    singleton_terms
        .try_reserve_exact(slots)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    singleton_terms.resize(slots, None);
    let mut has_constant = false;
    for term in &terms {
        let degree = term
            .mask
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if degree > 1 {
            return Err(ComputeFailure::NonAffine);
        }
        if degree == 0 {
            has_constant = true;
        } else {
            let slot = highest_set_bit(&term.mask).ok_or(ComputeFailure::Internal)?;
            let entry = singleton_terms
                .get_mut(slot)
                .ok_or(ComputeFailure::Internal)?;
            *entry = Some(term.matrix.as_slice());
        }
    }
    if !has_constant {
        return Err(ComputeFailure::Internal);
    }

    // I08 emits cut widths/live counts only. The reference Packet also forms
    // the target RHS and tests feasibility, but neither value is an I08 channel.
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(slots)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    for matrix in singleton_terms {
        let Some(matrix) = matrix else {
            columns.push(zeroed_u64(flat_words)?);
            continue;
        };
        columns.push(flatten_matrix(matrix, n, words_per_row, flat_words)?);
    }

    let prefix = incremental_ranks(&columns, false)?;
    let suffix = incremental_ranks(&columns, true)?;
    let total_rank = *prefix.last().ok_or(ComputeFailure::Internal)?;
    let groups = slots.checked_add(1).ok_or(ComputeFailure::ResourceLimit)?;
    let value_count = groups.checked_mul(4).ok_or(ComputeFailure::ResourceLimit)?;
    let incidence_count = groups
        .checked_mul(n)
        .and_then(|count| count.checked_mul(2))
        .ok_or(ComputeFailure::ResourceLimit)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    let mut incidence = zeroed_f32(incidence_count)?;
    let mut token_valid = Vec::new();
    token_valid
        .try_reserve_exact(groups)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    token_valid.resize(groups, 1);

    for cut in 0..groups {
        let width = prefix[cut]
            .checked_add(suffix[cut])
            .and_then(|sum| sum.checked_sub(total_rank))
            .ok_or(ComputeFailure::Internal)?;
        let mut live_slots = Vec::new();
        live_slots
            .try_reserve_exact(cut)
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        for slot in 0..cut {
            let live = context.factors.iter().any(|factor| {
                factor.scope.iter().copied().max().is_some_and(|last| {
                    usize::try_from(last).is_ok_and(|last| last >= cut)
                        && factor.scope.contains(&(slot as u32))
                })
            });
            if live {
                live_slots.push(slot);
                let (control, target) = context.gates[slot];
                let control = usize::try_from(control).map_err(|_| ComputeFailure::Internal)?;
                let target = usize::try_from(target).map_err(|_| ComputeFailure::Internal)?;
                incidence[(cut * n + control) * 2] = 1.0;
                incidence[(cut * n + target) * 2 + 1] = 1.0;
            }
        }
        let live = live_slots.len();
        let upper = width.checked_add(live).ok_or(ComputeFailure::Internal)?;
        if cut > MAX_EXACT_F32_INTEGER || upper > MAX_EXACT_F32_INTEGER {
            return Err(ComputeFailure::ResourceLimit);
        }
        values.extend([cut as f32, width as f32, live as f32, upper as f32]);
    }

    // Keep the state argument in the exact contract even though the displayed
    // widths are independent of its RHS.
    if state.dimension() != context.n {
        return Err(ComputeFailure::Internal);
    }
    Ok(FeatureValue {
        shape: vec![groups, 4],
        values,
        token_valid,
        incidence,
    })
}

fn working_bytes_upper_bound(n: usize, slots: usize, max_monomials: usize) -> Option<usize> {
    let words_per_row = n.checked_add(63)?.checked_div(64)?;
    let matrix_bytes = n
        .checked_mul(words_per_row)?
        .checked_mul(size_of::<u64>())?;
    let mask_bytes = words_for(slots)?.checked_mul(size_of::<u64>())?;
    let per_term = matrix_bytes
        .checked_add(mask_bytes)?
        .checked_add(size_of::<PolynomialTerm>())?
        .checked_add(128)?;
    let polynomial = max_monomials.checked_mul(per_term)?.checked_mul(2)?;
    let flat_words = words_for(n.checked_mul(n)?)?;
    let rank_workspace = slots
        .checked_mul(flat_words)?
        .checked_mul(size_of::<u64>())?
        .checked_mul(2)?;
    let groups = slots.checked_add(1)?;
    let output = groups
        .checked_mul(
            4usize
                .checked_add(n.checked_mul(2)?)?
                .checked_mul(size_of::<f32>())?,
        )?
        .checked_add(groups)?;
    polynomial.checked_add(rank_workspace)?.checked_add(output)
}

fn incremental_ranks(columns: &[Vec<u64>], reverse: bool) -> Result<Vec<usize>, ComputeFailure> {
    let mut ranks = Vec::new();
    ranks
        .try_reserve_exact(
            columns
                .len()
                .checked_add(1)
                .ok_or(ComputeFailure::ResourceLimit)?,
        )
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    ranks.resize(columns.len() + 1, 0);
    let mut pivots = BTreeMap::<usize, Vec<u64>>::new();
    if reverse {
        for index in (0..columns.len()).rev() {
            insert_basis(&mut pivots, &columns[index])?;
            ranks[index] = pivots.len();
        }
    } else {
        for index in 0..columns.len() {
            insert_basis(&mut pivots, &columns[index])?;
            ranks[index + 1] = pivots.len();
        }
    }
    Ok(ranks)
}

fn insert_basis(
    pivots: &mut BTreeMap<usize, Vec<u64>>,
    vector: &[u64],
) -> Result<(), ComputeFailure> {
    let mut row = Vec::new();
    row.try_reserve_exact(vector.len())
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    row.extend_from_slice(vector);
    for (pivot, basis_row) in pivots.iter().rev() {
        if bit(&row, *pivot) {
            xor_assign(&mut row, basis_row);
        }
    }
    let Some(pivot) = highest_set_bit(&row) else {
        return Ok(());
    };
    for basis_row in pivots.values_mut() {
        if bit(basis_row, pivot) {
            xor_assign(basis_row, &row);
        }
    }
    pivots.insert(pivot, row);
    Ok(())
}

fn flatten_matrix(
    matrix: &[u64],
    n: usize,
    words_per_row: usize,
    flat_words: usize,
) -> Result<Vec<u64>, ComputeFailure> {
    let mut flat = zeroed_u64(flat_words)?;
    for row in 0..n {
        for source_word in 0..words_per_row {
            let mut word = matrix[row * words_per_row + source_word];
            while word != 0 {
                let column = source_word * 64 + word.trailing_zeros() as usize;
                if column < n {
                    let bit_index = row
                        .checked_mul(n)
                        .and_then(|offset| offset.checked_add(column))
                        .ok_or(ComputeFailure::ResourceLimit)?;
                    set_bit(&mut flat, bit_index);
                }
                word &= word - 1;
            }
        }
    }
    Ok(flat)
}

fn words_for(width: usize) -> Option<usize> {
    width.checked_add(63).map(|value| value / 64)
}

fn zeroed_u64(length: usize) -> Result<Vec<u64>, ComputeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    values.resize(length, 0);
    Ok(values)
}

fn zeroed_f32(length: usize) -> Result<Vec<f32>, ComputeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn set_bit(words: &mut [u64], index: usize) {
    words[index / 64] |= 1u64 << (index % 64);
}

fn bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

fn highest_set_bit(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().rev().find_map(|(index, word)| {
        (*word != 0).then(|| index * 64 + (63 - word.leading_zeros() as usize))
    })
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= right;
    }
}

fn factor_entry_count(factors: &[LocalBooleanFactor]) -> Option<usize> {
    factors.iter().try_fold(0usize, |total, factor| {
        total
            .checked_add(factor.scope.len())?
            .checked_add(factor.allowed.len())
    })
}

fn producer_version(
    contexts: &[OptionalGatePacket],
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(
        &mut digest,
        b"stage59_optional_packet_v2_source_checked_cache",
    );
    hash_field(&mut digest, STAGE59_SOURCE_SHA256.as_bytes());
    for budget in [max_n, max_slots, max_monomials, max_factor_entries] {
        hash_field(&mut digest, &(budget as u64).to_le_bytes());
    }
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_field(&mut digest, &(context.gates.len() as u64).to_le_bytes());
        for (control, target) in &context.gates {
            hash_field(&mut digest, &control.to_le_bytes());
            hash_field(&mut digest, &target.to_le_bytes());
        }
        hash_field(&mut digest, &(context.factors.len() as u64).to_le_bytes());
        for factor in &context.factors {
            hash_field(&mut digest, &(factor.scope.len() as u64).to_le_bytes());
            for selector in &factor.scope {
                hash_field(&mut digest, &selector.to_le_bytes());
            }
            hash_field(&mut digest, &(factor.allowed.len() as u64).to_le_bytes());
            for pattern in &factor.allowed {
                hash_field(&mut digest, &pattern.to_bytes_le());
            }
        }
    }
    format!(
        "stage59_optional_packet_v2_source_checked_cache:{}",
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

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

/// Registers I08 for explicit, labelled optional-CNOT contexts.
pub fn register_positive_packet_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<OptionalGatePacket>,
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty()
        || max_n == 0
        || max_slots == 0
        || max_monomials == 0
        || max_factor_entries == 0
        || max_slots > MAX_EXACT_F32_INTEGER / 2
    {
        return Err(FeatureBuildError(
            "I08 requires contexts and positive budgets with exact float labels".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts
        .iter()
        .any(|context| !widths.insert(context.n) || factor_entry_count(&context.factors).is_none())
    {
        return Err(FeatureBuildError(
            "I08 allows one valid labelled context per physical width".into(),
        ));
    }
    let producer_version = producer_version(
        &contexts,
        max_n,
        max_slots,
        max_monomials,
        max_factor_entries,
    );
    let id = FeatureId::new(b'I', 8).expect("I08 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = 4;
    let spec = spec.with_provider_metadata(
        "fixed_order_affine_optional_CNOT_packet;explicit_local_factors;cut,kappa,live,log2_state_upper_bound;not_distance",
        producer_version,
        ["M", "optional_physical_gate_slots", "local_factor_truth_tables"],
        "exact",
        "T2",
        ["control", "target"],
        false,
    );
    registry.register(Box::new(PositivePacketProducer {
        spec,
        contexts,
        max_n,
        max_slots,
        max_monomials,
        max_factor_entries,
    }))
}

pub fn register_default_positive_packet_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<OptionalGatePacket>,
) -> Result<(), FeatureBuildError> {
    register_positive_packet_contexts(
        registry,
        contexts,
        DEFAULT_MAX_N,
        DEFAULT_MAX_SLOTS,
        DEFAULT_MAX_MONOMIALS,
        DEFAULT_MAX_FACTOR_ENTRIES,
    )
}
