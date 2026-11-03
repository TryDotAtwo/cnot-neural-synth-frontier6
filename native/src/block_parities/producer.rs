use super::BlockContext;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, feature_spec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ParityPattern(Vec<u64>);

impl ParityPattern {
    fn bit(&self, index: usize) -> bool {
        self.0
            .get(index / 64)
            .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    fn intersection_count(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(&other.0)
            .map(|(left, right)| (left & right).count_ones() as usize)
            .sum()
    }
}

impl Ord for ParityPattern {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for ParityPattern {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ParityData {
    codes: Vec<ParityPattern>,
    counts: HashMap<ParityPattern, usize>,
    columns: Vec<ParityPattern>,
}

struct PreparedBlock<'a> {
    context: &'a BlockContext,
    data: ParityData,
    patterns: Vec<ParityPattern>,
}

struct BlockParityProducer {
    spec: FeatureSpec,
    contexts: Vec<BlockContext>,
    max_patterns: usize,
    max_pairs: usize,
}

impl FeatureProducer for BlockParityProducer {
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
                "physical_block_parities_require_augmentation_only_symmetry",
            );
        }
        let relevant = self
            .contexts
            .iter()
            .filter(|context| context.n == state.dimension())
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_physical_block_context_for_matrix_width",
            );
        }
        if self.spec.id.ordinal() == 5 && relevant.len() != 1 {
            return unavailable(
                FeatureStatus::Unsupported,
                "node_block_frequency_requires_exactly_one_context",
            );
        }

        let mut prepared = Vec::new();
        if prepared.try_reserve_exact(relevant.len()).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "block context allocation failed",
            );
        }
        for context in relevant {
            let data = match parity_data(state, context, self.max_patterns) {
                Ok(data) => data,
                Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
            };
            let mut patterns = Vec::new();
            if patterns.try_reserve_exact(data.counts.len()).is_err() {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "pattern index allocation failed",
                );
            }
            patterns.extend(data.counts.keys().cloned());
            patterns.sort_unstable();
            prepared.push(PreparedBlock {
                context,
                data,
                patterns,
            });
        }

        if self.spec.id.ordinal() == 5 {
            return compute_node_frequencies(state, &prepared[0]);
        }
        compute_group_feature(
            state,
            self.spec.id.ordinal(),
            self.spec.channels,
            &self.spec.wire_roles,
            &prepared,
            self.max_pairs,
        )
    }
}

fn compute_node_frequencies(
    state: &MatrixState,
    prepared: &PreparedBlock<'_>,
) -> FeatureComputation {
    let n = state.dimension() as usize;
    let Some(value_len) = n.checked_mul(2) else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "node block feature size overflow",
        );
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(value_len).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "node block feature allocation failed",
        );
    }
    values.resize(value_len, 0.0f32);
    for (target, code) in prepared.context.targets.iter().zip(&prepared.data.codes) {
        let Some(count) = prepared.data.counts.get(code).copied() else {
            return unavailable(
                FeatureStatus::Unsupported,
                "target parity pattern is unindexed",
            );
        };
        let Some(count) = exact_f32_integer(count) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "target frequency exceeds exact f32 range",
            );
        };
        values[*target as usize * 2] = count;
    }

    let mut column_counts = HashMap::new();
    if column_counts
        .try_reserve(prepared.data.columns.len())
        .is_err()
    {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "column-frequency index allocation failed",
        );
    }
    for code in &prepared.data.columns {
        let entry = column_counts.entry(code).or_insert(0usize);
        let Some(next) = entry.checked_add(1) else {
            return unavailable(FeatureStatus::ResourceLimit, "column frequency overflow");
        };
        *entry = next;
    }
    for (source, code) in prepared.context.sources.iter().zip(&prepared.data.columns) {
        let Some(count) = column_counts.get(code).copied() else {
            return unavailable(
                FeatureStatus::Unsupported,
                "source parity column is unindexed",
            );
        };
        let Some(count) = exact_f32_integer(count) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column frequency exceeds exact f32 range",
            );
        };
        values[*source as usize * 2 + 1] = count;
    }
    FeatureComputation::Computed(FeatureValue {
        shape: vec![n, 2],
        values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}

fn compute_group_feature(
    state: &MatrixState,
    ordinal: u8,
    channels: usize,
    wire_roles: &[String],
    prepared: &[PreparedBlock<'_>],
    max_pairs: usize,
) -> FeatureComputation {
    let mut group_count = 0usize;
    for block in prepared {
        let count = match ordinal {
            1 | 2 | 4 => block.patterns.len(),
            6 => {
                let nonzero = block
                    .patterns
                    .iter()
                    .filter(|pattern| !pattern.is_zero())
                    .count();
                match nonzero
                    .checked_mul(nonzero.saturating_sub(1))
                    .map(|pairs| pairs / 2)
                {
                    Some(pairs) if pairs <= max_pairs => pairs,
                    Some(_) => {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "nonzero_pattern_pair_budget_exceeded",
                        );
                    }
                    None => {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "pattern pair count overflow",
                        );
                    }
                }
            }
            7 | 8 => 1,
            _ => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "unsupported block parity feature",
                );
            }
        };
        let Some(next) = group_count.checked_add(count) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "block feature group count overflow",
            );
        };
        group_count = next;
    }
    let n = state.dimension() as usize;
    let Some(value_count) = group_count.checked_mul(channels) else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block feature value size overflow",
        );
    };
    let Some(incidence_count) = group_count
        .checked_mul(n)
        .and_then(|size| size.checked_mul(wire_roles.len()))
    else {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block feature incidence size overflow",
        );
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(value_count).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block feature value allocation failed",
        );
    }
    let mut incidence = Vec::new();
    if incidence.try_reserve_exact(incidence_count).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block incidence allocation failed",
        );
    }
    let mut token_valid = Vec::new();
    if token_valid.try_reserve_exact(group_count).is_err() {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "block token allocation failed",
        );
    }

    for block in prepared {
        match ordinal {
            1 | 2 | 4 => {
                for pattern in &block.patterns {
                    let count = block.data.counts[pattern];
                    let value = match ordinal {
                        1 => count,
                        2 => 1,
                        4 if !pattern.is_zero() => count,
                        4 => 0,
                        _ => unreachable!(),
                    };
                    let Some(value) = exact_f32_integer(value) else {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "pattern frequency exceeds exact f32 range",
                        );
                    };
                    values.push(value);
                    append_incidence(
                        block.context,
                        &block.data.codes,
                        Some(pattern),
                        IncidenceColumn::MatchingTargets(pattern),
                        &mut incidence,
                    );
                    token_valid.push(1);
                }
            }
            6 => {
                let nonzero = block
                    .patterns
                    .iter()
                    .filter(|pattern| !pattern.is_zero())
                    .collect::<Vec<_>>();
                for left_index in 0..nonzero.len() {
                    for right_index in (left_index + 1)..nonzero.len() {
                        let left = nonzero[left_index];
                        let right = nonzero[right_index];
                        let Some(value) = exact_f32_integer(left.intersection_count(right)) else {
                            return unavailable(
                                FeatureStatus::ResourceLimit,
                                "pattern intersection exceeds exact f32 range",
                            );
                        };
                        values.push(value);
                        append_incidence(
                            block.context,
                            &block.data.codes,
                            Some(left),
                            IncidenceColumn::SourcePattern(right),
                            &mut incidence,
                        );
                        token_valid.push(1);
                    }
                }
            }
            7 => {
                let nonzero = block
                    .patterns
                    .iter()
                    .filter(|pattern| !pattern.is_zero())
                    .collect::<Vec<_>>();
                let rank = match rank_patterns(&nonzero, block.context.sources.len()) {
                    Ok(rank) => rank,
                    Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                };
                let total = nonzero.iter().try_fold(0usize, |sum, pattern| {
                    sum.checked_add(block.data.counts[*pattern])
                });
                let Some(total) = total else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "nonzero pattern count overflow",
                    );
                };
                let max_frequency = nonzero
                    .iter()
                    .map(|pattern| block.data.counts[*pattern])
                    .max()
                    .unwrap_or(0);
                let scalars = [
                    rank,
                    nonzero.len(),
                    total,
                    total - nonzero.len(),
                    max_frequency,
                ];
                for scalar in scalars {
                    let Some(value) = exact_f32_integer(scalar) else {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "block rank summary exceeds exact f32 range",
                        );
                    };
                    values.push(value);
                }
                append_incidence(
                    block.context,
                    &block.data.codes,
                    None,
                    IncidenceColumn::AllTargets,
                    &mut incidence,
                );
                token_valid.push(1);
            }
            8 => {
                let parity_weight = block
                    .data
                    .codes
                    .iter()
                    .flat_map(|pattern| &pattern.0)
                    .map(|word| u128::from(word.count_ones()))
                    .sum::<u128>();
                let Some(value) = exact_f32_integer_u128(parity_weight) else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "block parity weight exceeds exact f32 range",
                    );
                };
                values.push(value);
                append_incidence(
                    block.context,
                    &block.data.codes,
                    None,
                    IncidenceColumn::AllTargets,
                    &mut incidence,
                );
                token_valid.push(1);
            }
            _ => unreachable!(),
        }
    }
    FeatureComputation::Computed(FeatureValue {
        shape: vec![group_count, channels],
        values,
        token_valid,
        incidence,
    })
}

enum IncidenceColumn<'a> {
    SourcePattern(&'a ParityPattern),
    MatchingTargets(&'a ParityPattern),
    AllTargets,
}

fn append_incidence(
    context: &BlockContext,
    codes: &[ParityPattern],
    role_one_pattern: Option<&ParityPattern>,
    role_two: IncidenceColumn<'_>,
    output: &mut Vec<f32>,
) {
    for wire in 0..context.n {
        let source_position = context.sources.binary_search(&wire).ok();
        output.push(f32::from(source_position.is_some()));
        let role_one = role_one_pattern
            .and_then(|pattern| source_position.map(|index| pattern.bit(index)))
            .unwrap_or_else(|| role_one_pattern.is_none() && source_position.is_some());
        output.push(f32::from(role_one));
        let role_two = match &role_two {
            IncidenceColumn::SourcePattern(pattern) => source_position
                .map(|index| pattern.bit(index))
                .unwrap_or(false),
            IncidenceColumn::MatchingTargets(pattern) => context
                .targets
                .binary_search(&wire)
                .ok()
                .and_then(|index| codes.get(index))
                .is_some_and(|code| code == *pattern),
            IncidenceColumn::AllTargets => context.targets.binary_search(&wire).is_ok(),
        };
        output.push(f32::from(role_two));
    }
}

fn rank_patterns(patterns: &[&ParityPattern], source_count: usize) -> Result<usize, &'static str> {
    let words = source_count
        .checked_add(63)
        .ok_or("parity matrix width overflow")?
        / 64;
    let word_count = patterns
        .len()
        .checked_mul(words)
        .ok_or("parity matrix storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "parity matrix allocation failed")?;
    for pattern in patterns {
        matrix.extend_from_slice(&pattern.0);
    }
    Ok(eliminate_packed_rows(
        &mut matrix,
        patterns.len(),
        source_count,
        words,
    ))
}

fn parity_data(
    state: &MatrixState,
    context: &BlockContext,
    max_patterns: usize,
) -> Result<ParityData, &'static str> {
    let source_count = context.sources.len();
    let target_count = context.targets.len();
    let source_words = source_count
        .checked_add(63)
        .ok_or("source parity width overflow")?
        / 64;
    let target_words = target_count
        .checked_add(63)
        .ok_or("target parity width overflow")?
        / 64;
    let zero = ParityPattern(zeroed_words(source_words)?);
    let mut codes = Vec::new();
    codes
        .try_reserve_exact(target_count)
        .map_err(|_| "target parity allocation failed")?;
    let mut counts = HashMap::new();
    counts
        .try_reserve(
            target_count
                .checked_add(1)
                .ok_or("pattern count overflow")?
                .min(max_patterns),
        )
        .map_err(|_| "pattern count allocation failed")?;
    counts.insert(zero.clone(), 0usize);

    for target in &context.targets {
        let mut words = zeroed_words(source_words)?;
        for (source_index, source) in context.sources.iter().copied().enumerate() {
            if matrix_bit(state, *target, source)? {
                words[source_index / 64] |= 1u64 << (source_index % 64);
            }
        }
        let code = ParityPattern(words);
        if !counts.contains_key(&code) && counts.len() >= max_patterns {
            return Err("distinct parity pattern budget exceeded");
        }
        let count = counts.entry(code.clone()).or_insert(0);
        *count = count.checked_add(1).ok_or("pattern frequency overflow")?;
        codes.push(code);
    }

    let mut columns = Vec::new();
    columns
        .try_reserve_exact(source_count)
        .map_err(|_| "source parity-column allocation failed")?;
    for source_index in 0..source_count {
        let mut words = zeroed_words(target_words)?;
        for (target_index, code) in codes.iter().enumerate() {
            if code.bit(source_index) {
                words[target_index / 64] |= 1u64 << (target_index % 64);
            }
        }
        columns.push(ParityPattern(words));
    }
    Ok(ParityData {
        codes,
        counts,
        columns,
    })
}

fn matrix_bit(state: &MatrixState, row: u32, column: u32) -> Result<bool, &'static str> {
    let index = (row as usize)
        .checked_mul(state.words_per_row())
        .and_then(|base| base.checked_add(column as usize / 64))
        .ok_or("matrix parity index overflow")?;
    let word = state
        .packed_rows()
        .get(index)
        .ok_or("matrix parity index outside packed state")?;
    Ok(word & (1u64 << (column % 64)) != 0)
}

fn zeroed_words(count: usize) -> Result<Vec<u64>, &'static str> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(count)
        .map_err(|_| "parity workspace allocation failed")?;
    words.resize(count, 0);
    Ok(words)
}

fn eliminate_packed_rows(
    matrix: &mut [u64],
    row_count: usize,
    columns: usize,
    words: usize,
) -> usize {
    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (rank..row_count).find(|row| matrix[*row * words + word] & bit != 0)
        else {
            continue;
        };
        if pivot != rank {
            for index in 0..words {
                matrix.swap(pivot * words + index, rank * words + index);
            }
        }
        for row in (rank + 1)..row_count {
            if matrix[row * words + word] & bit != 0 {
                for index in 0..words {
                    let pivot_value = matrix[rank * words + index];
                    matrix[row * words + index] ^= pivot_value;
                }
            }
        }
        rank += 1;
        if rank == row_count {
            break;
        }
    }
    rank
}

fn exact_f32_integer(value: usize) -> Option<f32> {
    exact_f32_integer_u128(value as u128)
}

fn exact_f32_integer_u128(value: u128) -> Option<f32> {
    (value <= (1 << 24)).then_some(value as f32)
}

pub(super) fn valid_indices(values: &[u32], n: u32) -> bool {
    !values.is_empty()
        && values.iter().all(|value| *value < n)
        && !values.windows(2).any(|pair| pair[0] >= pair[1])
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

/// Registers D01/D02/D04-D08 for explicit physical source/target wire blocks.
pub fn register_block_parities(
    registry: &mut FeatureRegistry,
    contexts: Vec<BlockContext>,
    max_patterns: usize,
    max_pairs: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || max_patterns == 0 {
        return Err(FeatureBuildError(
            "explicit block contexts and a positive pattern budget are required".into(),
        ));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(contexts.len())
        .map_err(|_| FeatureBuildError("block-context validation allocation failed".into()))?;
    if contexts.iter().any(|context| !seen.insert(context)) {
        return Err(FeatureBuildError("block contexts must be unique".into()));
    }

    let ids = [1, 2, 4, 5, 6, 7, 8].map(|ordinal| {
        FeatureId::new(b'D', ordinal).expect("D01-D08 IDs are in the canonical feature manifest")
    });
    let specs = registry.specs();
    for id in &ids {
        if specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let mut digest = Sha256::new();
    digest.update(b"cnot103/physical-block-parities/v1\0");
    digest.update((contexts.len() as u64).to_le_bytes());
    for context in &contexts {
        digest.update(context.n.to_le_bytes());
        hash_indices(&mut digest, &context.sources);
        hash_indices(&mut digest, &context.targets);
    }
    digest.update((max_patterns as u64).to_le_bytes());
    digest.update((max_pairs as u64).to_le_bytes());
    let version = format!("physical_block_parities_v1:{}", finish_digest(digest));

    for id in ids {
        let ordinal = id.ordinal();
        let mut spec = feature_spec(id);
        spec.scope = "explicit_physical_block;sparse_exact_histogram;augmentation_only".into();
        spec.producer_version = version.clone();
        spec.dependencies = vec!["M".into(), "physical_source_target_context".into()];
        spec.exactness = "exact".into();
        spec.cost_tier = "T1".into();
        spec.wire_roles = if ordinal == 5 {
            Vec::new()
        } else if ordinal == 6 {
            vec![
                "source_coordinates".into(),
                "left_parity".into(),
                "right_parity".into(),
            ]
        } else {
            vec![
                "source_coordinates".into(),
                "parity_support".into(),
                "target_members".into(),
            ]
        };
        debug_assert!(matches!(spec.role, FeatureRole::Node | FeatureRole::Group));
        registry.register(Box::new(BlockParityProducer {
            spec,
            contexts: contexts.clone(),
            max_patterns,
            max_pairs,
        }))?;
    }
    Ok(())
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    digest.update((indices.len() as u64).to_le_bytes());
    for index in indices {
        digest.update(index.to_le_bytes());
    }
}

fn finish_digest(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
