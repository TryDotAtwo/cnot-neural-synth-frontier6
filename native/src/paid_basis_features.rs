//! T2 paid-basis feature producers D09-D14.
//!
//! These features are available only for explicitly supplied physical blocks.
//! Every value comes from the same exact-local candidate; the complete CNOT
//! word is replayed over all physical wires before endpoint features are emitted.
//! The endpoint digest is retained as packet provenance, never as a model value.

use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::paid_basis::{
    cancel_adjacent_block_gates, compile_paid_shear, endpoint_sha256, BasisSide,
    BoundaryCancellation, PaidBasisError, PaidShearCandidate, PaidShearConfig,
};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::{CnotGate, MatrixState};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

const CACHE_CAPACITY: usize = 16;
const MAX_CONTEXTS: usize = 1024;
const WIRE_ROLES: [&str; 2] = ["source_wire_set", "target_wire_set"];
const D_IDS: [u8; 6] = [9, 10, 11, 12, 13, 14];

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

struct PaidBasisFeatureProducer {
    id: FeatureId,
    spec: FeatureSpec,
    shared: Arc<PaidBasisFeatureCache>,
}

struct PaidBasisFeatureCache {
    contexts: Vec<PaidBasisBlockContext>,
    config: PaidShearConfig,
    entries: Mutex<VecDeque<(StateIdentity, CachedPaidBasisFeatures)>>,
}

#[derive(Clone)]
enum CachedPaidBasisFeatures {
    Computed(PaidBasisFeatureValues),
    Unavailable(FeatureStatus, String),
}

#[derive(Clone)]
struct PaidBasisFeatureValues {
    d09: FeatureValue,
    d10: FeatureValue,
    d11: FeatureValue,
    d12: FeatureValue,
    d13: FeatureValue,
    d14: FeatureValue,
    d14_provenance: Vec<String>,
}

impl PaidBasisFeatureValues {
    fn for_id(&self, id: FeatureId) -> FeatureComputation {
        match id.ordinal() {
            9 => FeatureComputation::Computed(self.d09.clone()),
            10 => FeatureComputation::Computed(self.d10.clone()),
            11 => FeatureComputation::Computed(self.d11.clone()),
            12 => FeatureComputation::Computed(self.d12.clone()),
            13 => FeatureComputation::Computed(self.d13.clone()),
            14 => FeatureComputation::ComputedWithProvenance {
                value: self.d14.clone(),
                provenance: self.d14_provenance.clone(),
            },
            _ => unreachable!("only D09-D14 share this producer cache"),
        }
    }
}

impl PaidBasisFeatureCache {
    fn get(&self, state: &MatrixState, policy: &FeaturePolicy) -> CachedPaidBasisFeatures {
        let key = StateIdentity::from(state);
        let Ok(mut entries) = self.entries.lock() else {
            return CachedPaidBasisFeatures::Unavailable(
                FeatureStatus::ResourceLimit,
                "paid_basis_cache_poisoned".into(),
            );
        };
        if let Some(index) = entries.iter().position(|(cached, _)| cached == &key) {
            let entry = entries
                .remove(index)
                .expect("located paid-basis cache entry");
            let result = entry.1.clone();
            entries.push_back(entry);
            return result;
        }
        let result = compute_paid_basis_features(state, policy, &self.contexts, self.config);
        if entries.len() == CACHE_CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, result.clone()));
        result
    }
}

impl FeatureProducer for PaidBasisFeatureProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != self.symmetry_contract() {
            return unavailable(
                FeatureStatus::Unsupported,
                "paid_basis_requires_augmentation_only_symmetry",
            );
        }
        match self.shared.get(state, policy) {
            CachedPaidBasisFeatures::Computed(values) => values.for_id(self.id),
            CachedPaidBasisFeatures::Unavailable(status, reason) => unavailable(status, reason),
        }
    }
}

/// Registers the six solver-assisted paid-basis features for explicit physical
/// blocks. Group features carry source/target incidence; D14 is a pair tensor
/// and exposes its endpoint digest only through packet provenance.
pub fn register_paid_basis_features(
    registry: &mut FeatureRegistry,
    contexts: Vec<PaidBasisBlockContext>,
    config: PaidShearConfig,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || contexts.len() > MAX_CONTEXTS {
        return Err(FeatureBuildError(format!(
            "paid-basis registration requires 1..={MAX_CONTEXTS} explicit contexts"
        )));
    }
    if !(2..=4).contains(&config.local_width) || config.sweeps > 16 {
        return Err(FeatureBuildError(
            "paid-basis local width must be 2..=4 and sweep count at most 16".into(),
        ));
    }
    let mut seen = HashSet::new();
    if contexts.iter().any(|context| !seen.insert(context.clone())) {
        return Err(FeatureBuildError(
            "paid-basis physical contexts must be unique".into(),
        ));
    }

    let specs = registry.specs();
    for ordinal in D_IDS {
        let id = FeatureId::new(b'D', ordinal).expect("D09-D14 are canonical feature IDs");
        if specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let version = format!(
        "verified_paid_basis_d09_d14_v1:{}",
        config_digest(&contexts, config)
    );
    let shared = Arc::new(PaidBasisFeatureCache {
        contexts,
        config,
        entries: Mutex::new(VecDeque::new()),
    });
    for ordinal in D_IDS {
        let id = FeatureId::new(b'D', ordinal).expect("D09-D14 are canonical feature IDs");
        let mut spec = feature_spec(id);
        spec.role = if ordinal == 14 {
            FeatureRole::Pair
        } else {
            FeatureRole::Group
        };
        spec.channels = d_channels(ordinal);
        spec.scope =
            "explicit_physical_blocks;verified_paid_basis_candidate;augmentation_only".into();
        spec.producer_version = version.clone();
        spec.dependencies = vec![
            "M".into(),
            "explicit_physical_blocks".into(),
            "exact_local_GL4_atlas".into(),
            "full_physical_CNOT_word_verifier".into(),
        ];
        spec.exactness =
            "exact_local_candidate;full_endpoint_replayed;not_global_optimality".into();
        spec.cost_tier = "T2".into();
        spec.wire_roles = if ordinal == 14 {
            Vec::new()
        } else {
            WIRE_ROLES.iter().map(|role| (*role).into()).collect()
        };
        spec.requires_solver_assisted = true;
        registry.register(Box::new(PaidBasisFeatureProducer {
            id,
            spec,
            shared: Arc::clone(&shared),
        }))?;
    }
    Ok(())
}

fn compute_paid_basis_features(
    state: &MatrixState,
    policy: &FeaturePolicy,
    all_contexts: &[PaidBasisBlockContext],
    config: PaidShearConfig,
) -> CachedPaidBasisFeatures {
    let fail = |status, reason: String| CachedPaidBasisFeatures::Unavailable(status, reason);
    let dimension = state.dimension();
    let relevant = all_contexts
        .iter()
        .filter(|context| context.dimension == dimension)
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return fail(
            FeatureStatus::Unsupported,
            "no_paid_basis_context_for_matrix_dimension".into(),
        );
    }
    let alphabet = match CnotAlphabet::all_to_all_unit(dimension) {
        Ok(alphabet) if alphabet.id() == state.alphabet_id() => alphabet,
        _ => {
            return fail(
                FeatureStatus::Unsupported,
                "paid_basis_requires_all_to_all_unit_CNOT_alphabet".into(),
            )
        }
    };
    let Some(estimated_bytes) = workspace_estimate(dimension as usize, relevant.len()) else {
        return fail(
            FeatureStatus::ResourceLimit,
            "paid_basis_workspace_size_overflow".into(),
        );
    };
    if policy
        .max_bytes
        .is_some_and(|limit| estimated_bytes > limit)
    {
        return fail(
            FeatureStatus::ResourceLimit,
            "paid_basis_workspace_budget_exceeded".into(),
        );
    }

    let mut candidates = Vec::new();
    if candidates.try_reserve_exact(relevant.len()).is_err() {
        return fail(
            FeatureStatus::ResourceLimit,
            "paid_basis_candidate_allocation_failed".into(),
        );
    }
    for context in &relevant {
        let block = match extract_block(state, context) {
            Ok(block) => block,
            Err(reason) => return fail(FeatureStatus::ResourceLimit, reason),
        };
        match compile_paid_shear(
            &block,
            dimension as usize,
            &context.source_wires,
            &context.target_wires,
            config,
        ) {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => return fail(status_for_compile_error(&error), error.to_string()),
        }
    }

    match assemble_feature_values(state, &alphabet, &relevant, &candidates) {
        Ok(values) => CachedPaidBasisFeatures::Computed(values),
        Err((status, reason)) => fail(status, reason),
    }
}

fn assemble_feature_values(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    contexts: &[&PaidBasisBlockContext],
    candidates: &[PaidShearCandidate],
) -> Result<PaidBasisFeatureValues, (FeatureStatus, String)> {
    let n = state.dimension() as usize;
    let group_count = candidates.len();
    let incidence_count = group_count
        .checked_mul(n)
        .and_then(|value| value.checked_mul(WIRE_ROLES.len()))
        .ok_or_else(|| resource("paid_basis_incidence_size_overflow"))?;
    let mut incidence = Vec::new();
    incidence
        .try_reserve_exact(incidence_count)
        .map_err(|_| resource("paid_basis_incidence_allocation_failed"))?;
    for context in contexts {
        for role in [&context.source_wires, &context.target_wires] {
            let mut membership = vec![0.0f32; n];
            for wire in role {
                membership[*wire] = 1.0;
            }
            incidence.extend(membership);
        }
    }
    let token_valid = vec![1u8; group_count];

    let mut d09 = Vec::with_capacity(group_count);
    let mut d10 = Vec::with_capacity(group_count);
    let mut d11 = Vec::with_capacity(group_count * 2);
    let mut d12 = Vec::with_capacity(group_count * 2);
    let mut d13 = Vec::with_capacity(group_count * 12);
    for candidate in candidates {
        d09.push(exact_count(candidate.source_preparation.len())?);
        d10.push(exact_count(candidate.target_preparation.len())?);
        d11.extend([
            exact_count(candidate.core.len())?,
            exact_count(candidate.restoration.len())?,
        ]);
        d12.extend([
            exact_count(candidate.paid_cost)?,
            exact_count(candidate.direct_cost.saturating_sub(candidate.paid_cost))?,
        ]);

        let source_basis_changes = changed_from_identity(&candidate.source_basis)?;
        let target_basis_changes = changed_from_identity(&candidate.target_basis)?;
        let source_moves = candidate
            .accepted_moves
            .iter()
            .filter(|movement| movement.side == BasisSide::Source)
            .count();
        let target_moves = candidate
            .accepted_moves
            .iter()
            .filter(|movement| movement.side == BasisSide::Target)
            .count();
        let swaps = physical_swap_count(&candidate.certificate.gates);
        let unshortened = candidate
            .preparation
            .len()
            .checked_mul(2)
            .and_then(|value| value.checked_add(candidate.core.len()))
            .ok_or_else(|| resource("paid_basis_unshortened_cost_overflow"))?;
        let internal_cancellations = sum_cancellations(&candidate.boundary_cancellations)?;
        d13.extend([
            exact_count(candidate.source_preparation.len())?,
            exact_count(source_basis_changes)?,
            exact_count(source_moves)?,
            exact_count(candidate.target_preparation.len())?,
            exact_count(target_basis_changes)?,
            exact_count(target_moves)?,
            exact_count(swaps)?,
            exact_count(
                swaps
                    .checked_mul(3)
                    .ok_or_else(|| resource("paid_basis_swap_price_overflow"))?,
            )?,
            exact_count(candidate.direct_cost)?,
            exact_count(unshortened)?,
            exact_count(internal_cancellations)?,
            exact_count(candidate.paid_cost)?,
        ]);
    }

    let d09 = group_value(1, group_count, d09, token_valid.clone(), incidence.clone());
    let d10 = group_value(1, group_count, d10, token_valid.clone(), incidence.clone());
    let d11 = group_value(2, group_count, d11, token_valid.clone(), incidence.clone());
    let d12 = group_value(2, group_count, d12, token_valid.clone(), incidence.clone());
    let d13 = group_value(12, group_count, d13, token_valid, incidence);

    let word_blocks = candidates
        .iter()
        .map(|candidate| candidate.certificate.gates.clone())
        .collect::<Vec<_>>();
    let compatibility = adjacent_compatibility_counts(&word_blocks);
    let reduced = cancel_adjacent_block_gates(&word_blocks);
    let start = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|error| unsupported(error.to_string()))?;
    let mut endpoint = start.clone();
    for gate in &reduced.word {
        endpoint
            .apply_cnot(*gate)
            .map_err(|error| unsupported(error.to_string()))?;
    }
    let combined_certificate = WordCertificate {
        start,
        end: endpoint.clone(),
        physical_cost: u64::try_from(reduced.word.len())
            .map_err(|_| resource("paid_basis_combined_word_length_exceeds_u64"))?,
        gates: reduced.word.clone(),
        producer: "verified_paid_basis_context_sequence_v1".into(),
        lineage_id: "explicit_paid_basis_contexts".into(),
    };
    combined_certificate
        .verify(alphabet)
        .map_err(|error| unsupported(error.to_string()))?;
    let endpoint_digest = endpoint_sha256(&endpoint);

    let pair_count = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| resource("paid_basis_d14_size_overflow"))?;
    let mut pair_values = Vec::new();
    pair_values
        .try_reserve_exact(pair_count)
        .map_err(|_| resource("paid_basis_d14_allocation_failed"))?;
    pair_values.resize(pair_count, 0.0f32);
    let mut gate_counts = HashMap::<(u32, u32), usize>::new();
    for gate in &reduced.word {
        let count = gate_counts.entry((gate.control, gate.target)).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| resource("paid_basis_d14_gate_count_overflow"))?;
    }
    let compatibility_counts = compatibility
        .into_iter()
        .map(|(gate, count)| (gate, count))
        .collect::<HashMap<_, _>>();
    let cancellation_counts = reduced
        .boundary_cancellations
        .iter()
        .map(|cancellation| {
            (
                (cancellation.control, cancellation.target),
                cancellation.canceled_pairs,
            )
        })
        .collect::<HashMap<_, _>>();
    for control in 0..n {
        for target in 0..n {
            let offset = (control * n + target) * 4;
            let endpoint_bit = matrix_bit(&endpoint, target, control);
            pair_values[offset] = f32::from(endpoint_bit);
            pair_values[offset + 1] = exact_count(
                gate_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
            pair_values[offset + 2] = exact_count(
                compatibility_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
            pair_values[offset + 3] = exact_count(
                cancellation_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
        }
    }
    let mut provenance = vec![format!("endpoint_sha256={endpoint_digest}")];
    for (index, candidate) in candidates.iter().enumerate() {
        provenance.push(format!(
            "block_{index}_endpoint_sha256={}",
            candidate.endpoint_sha256
        ));
    }
    let d14 = FeatureValue {
        shape: vec![n, n, 4],
        values: pair_values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    };

    Ok(PaidBasisFeatureValues {
        d09,
        d10,
        d11,
        d12,
        d13,
        d14,
        d14_provenance: provenance,
    })
}

fn extract_block(
    state: &MatrixState,
    context: &PaidBasisBlockContext,
) -> Result<Vec<Vec<u8>>, String> {
    let cells = context
        .source_wires
        .len()
        .checked_mul(context.target_wires.len())
        .ok_or_else(|| "paid_basis_block_size_overflow".to_owned())?;
    let mut block = Vec::new();
    block
        .try_reserve_exact(context.target_wires.len())
        .map_err(|_| "paid_basis_block_allocation_failed".to_owned())?;
    for &target in &context.target_wires {
        let mut row = Vec::new();
        row.try_reserve_exact(context.source_wires.len())
            .map_err(|_| "paid_basis_block_allocation_failed".to_owned())?;
        for &source in &context.source_wires {
            row.push(matrix_bit(state, target, source));
        }
        block.push(row);
    }
    if block.iter().map(Vec::len).sum::<usize>() != cells {
        return Err("paid_basis_block_extraction_mismatch".into());
    }
    Ok(block)
}

fn workspace_estimate(dimension: usize, groups: usize) -> Option<usize> {
    let pair_values = dimension.checked_mul(dimension)?.checked_mul(4)?;
    let pair_bytes = pair_values.checked_mul(std::mem::size_of::<f32>())?;
    let incidence = groups
        .checked_mul(dimension)?
        .checked_mul(WIRE_ROLES.len())?
        .checked_mul(std::mem::size_of::<f32>())?;
    let group_values = groups
        .checked_mul(19)?
        .checked_mul(std::mem::size_of::<f32>())?;
    pair_bytes.checked_add(incidence)?.checked_add(group_values)
}

fn d_channels(ordinal: u8) -> usize {
    match ordinal {
        9 | 10 => 1,
        11 | 12 => 2,
        13 => 12,
        14 => 4,
        _ => unreachable!("D09-D14 channel requested"),
    }
}

fn group_value(
    channels: usize,
    groups: usize,
    values: Vec<f32>,
    token_valid: Vec<u8>,
    incidence: Vec<f32>,
) -> FeatureValue {
    FeatureValue {
        shape: vec![groups, channels],
        values,
        token_valid,
        incidence,
    }
}

fn changed_from_identity(matrix: &[Vec<u8>]) -> Result<usize, (FeatureStatus, String)> {
    if matrix.iter().any(|row| row.len() != matrix.len()) {
        return Err(unsupported("paid_basis_matrix_not_square"));
    }
    Ok(matrix
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.iter()
                .enumerate()
                .filter(|(column, value)| **value != u8::from(row_index == *column))
                .count()
        })
        .sum())
}

fn physical_swap_count(gates: &[CnotGate]) -> usize {
    let mut count = 0usize;
    let mut index = 0usize;
    while index + 2 < gates.len() {
        let first = gates[index];
        let middle = gates[index + 1];
        let last = gates[index + 2];
        let is_swap =
            first == last && first.control == middle.target && first.target == middle.control;
        if is_swap {
            count += 1;
            index += 3;
        } else {
            index += 1;
        }
    }
    count
}

fn adjacent_compatibility_counts(blocks: &[Vec<CnotGate>]) -> HashMap<(u32, u32), usize> {
    let mut counts = HashMap::new();
    for adjacent in blocks.windows(2) {
        for (left, right) in adjacent[0].iter().rev().zip(&adjacent[1]) {
            if left != right {
                break;
            }
            let count = counts.entry((left.control, left.target)).or_insert(0usize);
            *count += 1;
        }
    }
    counts
}

fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> u8 {
    let words = state
        .row_words(row as u32)
        .expect("row index is bounded by the matrix dimension");
    ((words[column / 64] >> (column % 64)) & 1) as u8
}

fn exact_count(value: usize) -> Result<f32, (FeatureStatus, String)> {
    if value > (1usize << 24) {
        return Err(resource("paid_basis_integer_exceeds_exact_f32_range"));
    }
    Ok(value as f32)
}

fn sum_cancellations(
    cancellations: &[BoundaryCancellation],
) -> Result<usize, (FeatureStatus, String)> {
    cancellations.iter().try_fold(0usize, |sum, cancellation| {
        sum.checked_add(cancellation.canceled_pairs)
            .ok_or_else(|| resource("paid_basis_boundary_cancellation_overflow"))
    })
}

fn status_for_compile_error(error: &PaidBasisError) -> FeatureStatus {
    let message = error.to_string();
    if ["overflow", "allocation", "exceeds", "exhausted"]
        .iter()
        .any(|needle| message.contains(needle))
    {
        FeatureStatus::ResourceLimit
    } else {
        FeatureStatus::Unsupported
    }
}

fn config_digest(contexts: &[PaidBasisBlockContext], config: PaidShearConfig) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"verified_paid_basis_d09_d14_v1");
    hash_field(&mut digest, &(config.local_width as u64).to_le_bytes());
    hash_field(&mut digest, &(config.sweeps as u64).to_le_bytes());
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.dimension.to_le_bytes());
        hash_indices(&mut digest, &context.source_wires);
        hash_indices(&mut digest, &context.target_wires);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hash_indices(digest: &mut Sha256, values: &[usize]) {
    hash_field(digest, &(values.len() as u64).to_le_bytes());
    for value in values {
        hash_field(digest, &(*value as u64).to_le_bytes());
    }
}

fn sorted_in_range(values: &[usize], dimension: u32) -> bool {
    !values.is_empty()
        && values.iter().all(|value| *value < dimension as usize)
        && !values.windows(2).any(|pair| pair[0] >= pair[1])
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn resource(reason: impl Into<String>) -> (FeatureStatus, String) {
    (FeatureStatus::ResourceLimit, reason.into())
}

fn unsupported(reason: impl Into<String>) -> (FeatureStatus, String) {
    (FeatureStatus::Unsupported, reason.into())
}
