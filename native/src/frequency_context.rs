//! Source-relative I07 rank of an explicitly labelled stage-52 interface.
//!
//! The Fourier frequency is supplied by the caller. This provider reports
//! `kappa` and the exact representation-size exponent `w*kappa`; it does not
//! enumerate a spectrum or infer a frequency from the matrix rank.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::sync::Mutex;

const STAGE52_PRIMARY_SHA256: &str =
    "44fa3c82aa7b5c59542cb4101c40b8f437fca1e7baf47f66a3c395627c49c465";
const DEFAULT_MAX_WIDTH: u32 = 144;
const DEFAULT_MAX_FREQUENCY_BITS: usize = 20_736;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrequencyContextError(pub String);

impl fmt::Display for FrequencyContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FrequencyContextError {}

/// Physical wire partition and one labelled central Fourier frequency.
///
/// `central_rows` is an `m x r` matrix over GF(2): row `j` corresponds to the
/// `j`th target wire, and bit `i` of that row corresponds to the `i`th source
/// wire. Rows use little-endian `u64` limbs, matching `MatrixState` bit order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrequencyContext {
    n: u32,
    sources: Vec<u32>,
    internal: Vec<u32>,
    targets: Vec<u32>,
    central_rows: Vec<Vec<u64>>,
}

impl FrequencyContext {
    pub fn new(
        n: u32,
        sources: Vec<u32>,
        internal: Vec<u32>,
        targets: Vec<u32>,
        central_rows: Vec<Vec<u64>>,
    ) -> Result<Self, FrequencyContextError> {
        let width = usize::try_from(n)
            .map_err(|_| FrequencyContextError("interface width overflow".into()))?;
        if width == 0 || sources.is_empty() || internal.is_empty() || targets.is_empty() {
            return Err(FrequencyContextError(
                "positive width and nonempty source/internal/target blocks required".into(),
            ));
        }

        let mut seen = Vec::new();
        seen.try_reserve_exact(width)
            .map_err(|_| FrequencyContextError("wire-partition allocation failed".into()))?;
        seen.resize(width, false);
        for wire in sources.iter().chain(&internal).chain(&targets) {
            let index = usize::try_from(*wire)
                .map_err(|_| FrequencyContextError("wire index overflow".into()))?;
            let Some(slot) = seen.get_mut(index) else {
                return Err(FrequencyContextError(
                    "interface wire is outside the matrix".into(),
                ));
            };
            if std::mem::replace(slot, true) {
                return Err(FrequencyContextError(
                    "interface blocks must partition physical wires without duplicates".into(),
                ));
            }
        }
        if seen.iter().any(|present| !present) {
            return Err(FrequencyContextError(
                "interface blocks must cover every physical wire".into(),
            ));
        }

        let source_words = sources
            .len()
            .checked_add(63)
            .ok_or_else(|| FrequencyContextError("central-frequency width overflow".into()))?
            / 64;
        if central_rows.len() != targets.len()
            || central_rows.iter().any(|row| row.len() != source_words)
        {
            return Err(FrequencyContextError(
                "central frequency must have one packed source-width row per target".into(),
            ));
        }
        let used_last_word_bits = sources.len() % 64;
        if used_last_word_bits != 0 {
            let allowed = (1u64 << used_last_word_bits) - 1;
            if central_rows
                .iter()
                .any(|row| row[source_words - 1] & !allowed != 0)
            {
                return Err(FrequencyContextError(
                    "central frequency has nonzero padding bits".into(),
                ));
            }
        }

        Ok(Self {
            n,
            sources,
            internal,
            targets,
            central_rows,
        })
    }
}

struct FrequencyContextProducer {
    spec: FeatureSpec,
    contexts: Vec<FrequencyContext>,
    max_width: u32,
    max_frequency_bits: usize,
    rank_cache: Mutex<std::collections::HashMap<u32, usize>>,
}

impl FeatureProducer for FrequencyContextProducer {
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
                "i07_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "i07_no_explicit_frequency_context_for_width",
            );
        };
        let expected_alphabet = format!("all_to_all_unit_cnot:{}", context.n);
        if state.alphabet_id() != expected_alphabet || state.dimension() != context.n {
            return unavailable(FeatureStatus::Unsupported, "i07_state_alphabet_mismatch");
        }

        let Some(frequency_bits) = context.sources.len().checked_mul(context.targets.len()) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_frequency_bit_count_overflow",
            );
        };
        if context.n > self.max_width || frequency_bits > self.max_frequency_bits {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_explicit_context_exceeds_configured_budget",
            );
        }
        match has_stage52_interface(state, context) {
            Ok(true) => {}
            Ok(false) => {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "i07_state_outside_explicit_stage52_interface",
                );
            }
            Err(()) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "i07_interface_rank_allocation_failed",
                );
            }
        }

        let Some(kappa) = self.rank_for(context) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_central_rank_allocation_failed",
            );
        };
        let Some(representation_exponent) = context.internal.len().checked_mul(kappa) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_representation_exponent_overflow",
            );
        };
        if kappa > MAX_EXACT_F32_INTEGER || representation_exponent > MAX_EXACT_F32_INTEGER {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_feature_value_exceeds_exact_float32_integer_range",
            );
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![2],
            values: vec![kappa as f32, representation_exponent as f32],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl FrequencyContextProducer {
    fn rank_for(&self, context: &FrequencyContext) -> Option<usize> {
        match self.rank_cache.lock() {
            Ok(mut cache) => {
                if let Some(rank) = cache.get(&context.n) {
                    return Some(*rank);
                }
                let rank = gf2_rank(&context.central_rows, context.sources.len())?;
                cache.insert(context.n, rank);
                Some(rank)
            }
            Err(_) => gf2_rank(&context.central_rows, context.sources.len()),
        }
    }
}

/// Registers I07 for explicit labelled interface contexts, with source budgets.
///
/// The default limits match the Python reference's width and `r*m` frequency
/// budget. Contexts for distinct widths are required; their wire order is kept
/// exactly as supplied and is not relabel-invariant.
pub fn register_frequency_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<FrequencyContext>,
    max_width: u32,
    max_frequency_bits: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || max_width == 0 || max_frequency_bits == 0 {
        return Err(FeatureBuildError(
            "I07 requires explicit contexts and positive width/frequency budgets".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I07 allows at most one labelled Fourier context per width".into(),
        ));
    }

    let producer_version = producer_version(&contexts, max_width, max_frequency_bits);
    let id = FeatureId::new(b'I', 7).expect("I07 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 2;
    let spec = spec.with_provider_metadata(
        concat!(
            "explicit_full_stage52_interface;fixed_labelled_central_Fourier_sector;",
            "kappa_and_log2_representation_dimension;not_distance"
        ),
        producer_version,
        [
            "M",
            "physical_interface_partition",
            "central_frequency_matrix",
        ],
        "exact",
        "T2",
        std::iter::empty::<String>(),
        false,
    );
    registry.register(Box::new(FrequencyContextProducer {
        spec,
        contexts,
        max_width,
        max_frequency_bits,
        rank_cache: Mutex::new(std::collections::HashMap::new()),
    }))
}

pub fn register_default_frequency_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<FrequencyContext>,
) -> Result<(), FeatureBuildError> {
    register_frequency_contexts(
        registry,
        contexts,
        DEFAULT_MAX_WIDTH,
        DEFAULT_MAX_FREQUENCY_BITS,
    )
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn has_stage52_interface(state: &MatrixState, context: &FrequencyContext) -> Result<bool, ()> {
    let n = context.n;
    for (index, row) in context.sources.iter().copied().enumerate() {
        for column in 0..n {
            if state_bit(state, row, column) != (column == context.sources[index]) {
                return Ok(false);
            }
        }
    }
    for row in context.internal.iter().copied() {
        if context
            .targets
            .iter()
            .copied()
            .any(|column| state_bit(state, row, column))
        {
            return Ok(false);
        }
    }
    for (row_index, row) in context.targets.iter().copied().enumerate() {
        for (column_index, column) in context.targets.iter().copied().enumerate() {
            if state_bit(state, row, column) != (row_index == column_index) {
                return Ok(false);
            }
        }
    }

    let Some(internal_words) = context.internal.len().checked_add(63).map(|bits| bits / 64) else {
        return Err(());
    };
    let mut b_rows = Vec::new();
    if b_rows.try_reserve_exact(context.internal.len()).is_err() {
        return Err(());
    }
    for row in context.internal.iter().copied() {
        let mut packed = Vec::new();
        if packed.try_reserve_exact(internal_words).is_err() {
            return Err(());
        }
        packed.resize(internal_words, 0u64);
        for (column_index, column) in context.internal.iter().copied().enumerate() {
            if state_bit(state, row, column) {
                packed[column_index / 64] |= 1u64 << (column_index % 64);
            }
        }
        b_rows.push(packed);
    }
    Ok(gf2_rank(&b_rows, context.internal.len()).ok_or(())? == context.internal.len())
}

fn state_bit(state: &MatrixState, row: u32, column: u32) -> bool {
    let word = (column / 64) as usize;
    let bit = 1u64 << (column % 64);
    state
        .row_words(row)
        .and_then(|words| words.get(word))
        .is_some_and(|value| value & bit != 0)
}

fn gf2_rank(rows: &[Vec<u64>], columns: usize) -> Option<usize> {
    let word_count = columns.checked_add(63)? / 64;
    if rows.iter().any(|row| row.len() != word_count) {
        return None;
    }
    let mut work = Vec::new();
    work.try_reserve_exact(rows.len()).ok()?;
    for row in rows {
        let mut copy = Vec::new();
        copy.try_reserve_exact(row.len()).ok()?;
        copy.extend_from_slice(row);
        work.push(copy);
    }

    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (rank..work.len()).find(|row| work[*row][word] & bit != 0) else {
            continue;
        };
        work.swap(rank, pivot);
        for row in 0..work.len() {
            if row != rank && work[row][word] & bit != 0 {
                for index in 0..word_count {
                    work[row][index] ^= work[rank][index];
                }
            }
        }
        rank += 1;
        if rank == work.len() {
            break;
        }
    }
    Some(rank)
}

fn producer_version(
    contexts: &[FrequencyContext],
    max_width: u32,
    max_frequency_bits: usize,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/stage52-frequency-sector/v1\0");
    hash_bytes(&mut digest, STAGE52_PRIMARY_SHA256.as_bytes());
    hash_u64(&mut digest, max_width as u64);
    hash_u64(&mut digest, max_frequency_bits as u64);
    hash_u64(&mut digest, contexts.len() as u64);
    for context in contexts {
        hash_u64(&mut digest, context.n as u64);
        for block in [&context.sources, &context.internal, &context.targets] {
            hash_u64(&mut digest, block.len() as u64);
            for wire in block {
                hash_u64(&mut digest, *wire as u64);
            }
        }
        hash_u64(&mut digest, context.central_rows.len() as u64);
        for row in &context.central_rows {
            hash_u64(&mut digest, row.len() as u64);
            for word in row {
                hash_u64(&mut digest, *word);
            }
        }
    }
    format!("stage52_frequency_sector_v2:{}", hex(&digest.finalize()))
}

fn hash_bytes(digest: &mut Sha256, value: &[u8]) {
    hash_u64(digest, value.len() as u64);
    digest.update(value);
}

fn hash_u64(digest: &mut Sha256, value: u64) {
    digest.update(value.to_le_bytes());
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}
