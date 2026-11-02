//! Checked source port of the stage-26 shared-parity lower-bound features.
//!
//! F03 reports certificate dimensions, F04 reports projected coset relations,
//! and F05 reports the certified lower bound next to the elementary bound.
//! These are lower-bound diagnostics, never exact CNOT distances.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

pub const STAGE26_CERTIFICATE_SOURCE_SHA256: &str =
    "6d29d12d2eb369e120533e1961f4b984429643a08f93cc4db99b148a9088bc05";
pub const STAGE26_CHECKER_SOURCE_SHA256: &str =
    "4dfda79863fe7a7b57785e023471cc5d0e8dac21cc3313271cacf4bf97db08ce";
pub const STAGE26_PROOF_NOTES_SHA256: &str =
    "0daadfb3738478b6ee95c11aad4d67177a1c17613eba5e3c91dc041151676fea";
const DEFAULT_MAX_N: u32 = 144;
const DEFAULT_MAX_CHECKER_PAIRS: usize = 100_000;
const CACHE_CAPACITY: usize = 32;

const STAGE26_IDS: [(u8, u8, usize); 3] = [(b'F', 3, 5), (b'F', 4, 3), (b'F', 5, 2)];

#[derive(Clone, Debug, PartialEq)]
struct Stage26Values {
    f03: Vec<f32>,
    f04: Vec<f32>,
    f05: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Certificate {
    rows: Vec<BigUint>,
    retained: Vec<bool>,
    projected_rows: Vec<BigUint>,
    deleted_columns: BTreeSet<usize>,
    deletion_witness_rows: BTreeMap<usize, usize>,
    targets: Vec<BigUint>,
    h_basis: Vec<BigUint>,
    deletions: usize,
    target_count: usize,
    obstruction_rank: usize,
    h_dimension: usize,
    w0_dimension: usize,
    lower_bound: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum EvaluationError {
    UnsupportedAlphabet,
    ResourceLimit(&'static str),
    VerificationFailed(&'static str),
}

fn one_hot(index: usize) -> BigUint {
    BigUint::from(1u8) << index
}

fn row_value(state: &MatrixState, row: usize) -> BigUint {
    let mut value = BigUint::from(0u8);
    for (word, limb) in state
        .row_words(row as u32)
        .expect("row index is in range")
        .iter()
        .enumerate()
    {
        value |= BigUint::from(*limb) << (word * 64);
    }
    value
}

fn high_basis(values: impl IntoIterator<Item = BigUint>) -> Vec<BigUint> {
    let mut pivots = BTreeMap::<u64, BigUint>::new();
    for mut value in values {
        loop {
            if value == BigUint::from(0u8) {
                break;
            }
            let pivot = value.bits() - 1;
            if let Some(existing) = pivots.get(&pivot) {
                value ^= existing;
            } else {
                pivots.insert(pivot, value);
                break;
            }
        }
    }
    let positions: Vec<_> = pivots.keys().copied().collect();
    for low in &positions {
        for high in &positions {
            if high > low && pivots[high].bit(*low) {
                let low_value = pivots[low].clone();
                *pivots.get_mut(high).expect("pivot was collected") ^= low_value;
            }
        }
    }
    positions
        .into_iter()
        .rev()
        .map(|pivot| pivots.remove(&pivot).expect("pivot was collected"))
        .collect()
}

fn high_remainder(mut value: BigUint, basis: &[BigUint]) -> BigUint {
    for vector in basis {
        let pivot = vector.bits() - 1;
        if value.bit(pivot) {
            value ^= vector;
        }
    }
    value
}

fn lowest_set_bit(value: &BigUint) -> Option<u64> {
    (0..value.bits()).find(|bit| value.bit(*bit))
}

// The checker deliberately uses low pivots and pairwise coset differences;
// the constructor uses high pivots and one representative per coset.
fn low_basis(values: impl IntoIterator<Item = BigUint>) -> BTreeMap<u64, BigUint> {
    let mut pivots = BTreeMap::<u64, BigUint>::new();
    for mut value in values {
        loop {
            let Some(pivot) = lowest_set_bit(&value) else {
                break;
            };
            if let Some(existing) = pivots.get(&pivot) {
                value ^= existing;
            } else {
                pivots.insert(pivot, value);
                break;
            }
        }
    }
    pivots
}

fn low_remainder(mut value: BigUint, basis: &BTreeMap<u64, BigUint>) -> BigUint {
    loop {
        let Some(pivot) = lowest_set_bit(&value) else {
            return value;
        };
        let Some(vector) = basis.get(&pivot) else {
            return value;
        };
        value ^= vector;
    }
}

fn projected(value: &BigUint, retained: &[bool]) -> BigUint {
    let mut output = BigUint::from(0u8);
    for (column, keep) in retained.iter().copied().enumerate() {
        if keep && value.bit(column as u64) {
            output |= one_hot(column);
        }
    }
    output
}

fn make_certificate(state: &MatrixState) -> Certificate {
    let n = state.dimension() as usize;
    let rows: Vec<_> = (0..n).map(|row| row_value(state, row)).collect();
    let changed: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(row, value)| *value != one_hot(row))
        .collect();
    let mut off_diagonal_columns = vec![false; n];
    for (row, value) in rows.iter().enumerate() {
        for (column, marked) in off_diagonal_columns.iter_mut().enumerate() {
            if row != column && value.bit(column as u64) {
                *marked = true;
            }
        }
    }
    let retained: Vec<_> = (0..n)
        .map(|wire| !(changed[wire] && !off_diagonal_columns[wire]))
        .collect();
    let changing_retained: Vec<_> = (0..n).map(|wire| changed[wire] && retained[wire]).collect();
    let projected_rows: Vec<_> = rows.iter().map(|row| projected(row, &retained)).collect();

    let mut deleted_columns = BTreeSet::new();
    let mut deletion_witness_rows = BTreeMap::new();
    for (row_index, row) in rows.iter().enumerate() {
        let has_retained_part = retained
            .iter()
            .enumerate()
            .any(|(column, keep)| *keep && row.bit(column as u64));
        if !has_retained_part {
            continue;
        }
        for (column, keep) in retained.iter().enumerate() {
            if !keep && row.bit(column as u64) {
                deleted_columns.insert(column);
                deletion_witness_rows.entry(column).or_insert(row_index);
            }
        }
    }

    let inputs: BTreeSet<_> = retained
        .iter()
        .enumerate()
        .filter_map(|(wire, keep)| keep.then(|| one_hot(wire)))
        .collect();
    let targets: Vec<_> = projected_rows
        .iter()
        .filter(|value| **value != BigUint::from(0u8) && !inputs.contains(*value))
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let changing_support = rows
        .iter()
        .filter(|value| {
            changing_retained
                .iter()
                .enumerate()
                .any(|(wire, active)| *active && value.bit(wire as u64))
        })
        .map(|value| projected(value, &retained));
    let h_basis = high_basis(changing_support);
    let mut representatives = BTreeMap::<BigUint, BigUint>::new();
    let mut w0_generators = Vec::new();
    for value in inputs
        .iter()
        .chain(&targets)
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        let coset = high_remainder(value.clone(), &h_basis);
        if coset == BigUint::from(0u8) {
            if inputs.contains(&value) {
                w0_generators.push(value);
            }
        } else if let Some(representative) = representatives.get(&coset) {
            w0_generators.push(value ^ representative);
        } else {
            representatives.insert(coset, value);
        }
    }
    let w0_basis = high_basis(w0_generators);
    let targets_in_h: Vec<_> = targets
        .iter()
        .filter(|target| high_remainder((*target).clone(), &h_basis) == BigUint::from(0u8))
        .cloned()
        .collect();
    let mut extended_generators = w0_basis.clone();
    extended_generators.extend(targets_in_h.iter().cloned());
    let extended_basis = high_basis(extended_generators);
    let obstruction_rank = extended_basis.len() - w0_basis.len();
    let lower_bound = deleted_columns.len() + targets.len() + obstruction_rank;
    let certificate = Certificate {
        rows,
        retained,
        projected_rows,
        deleted_columns,
        deletion_witness_rows,
        targets,
        h_dimension: h_basis.len(),
        h_basis,
        w0_dimension: w0_basis.len(),
        obstruction_rank,
        deletions: 0,
        target_count: 0,
        lower_bound,
    };
    Certificate {
        deletions: certificate.deleted_columns.len(),
        target_count: certificate.targets.len(),
        ..certificate
    }
}

fn independently_verify(state: &MatrixState, certificate: &Certificate) -> bool {
    let n = state.dimension() as usize;
    if certificate.rows.len() != n || certificate.retained.len() != n {
        return false;
    }
    let source_rows: Vec<_> = (0..n).map(|row| row_value(state, row)).collect();
    if source_rows != certificate.rows
        || certificate
            .rows
            .iter()
            .map(|row| projected(row, &certificate.retained))
            .collect::<Vec<_>>()
            != certificate.projected_rows
    {
        return false;
    }
    let mut deleted = BTreeSet::new();
    for (row_index, row) in certificate.rows.iter().enumerate() {
        let has_retained_part = certificate
            .retained
            .iter()
            .enumerate()
            .any(|(column, keep)| *keep && row.bit(column as u64));
        if has_retained_part {
            for (column, keep) in certificate.retained.iter().enumerate() {
                if !keep && row.bit(column as u64) {
                    deleted.insert(column);
                    if certificate.deletion_witness_rows.get(&column) != Some(&row_index) {
                        // A valid witness may be any row witnessing this column.
                        if !certificate
                            .deletion_witness_rows
                            .get(&column)
                            .is_some_and(|witness| {
                                *witness < n
                                    && certificate.rows[*witness].bit(column as u64)
                                    && certificate.retained.iter().enumerate().any(
                                        |(wire, keep)| {
                                            *keep && certificate.rows[*witness].bit(wire as u64)
                                        },
                                    )
                            })
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    if deleted != certificate.deleted_columns || deleted.len() != certificate.deletions {
        return false;
    }
    if certificate
        .deletion_witness_rows
        .keys()
        .copied()
        .collect::<BTreeSet<_>>()
        != deleted
    {
        return false;
    }

    let inputs: BTreeSet<_> = certificate
        .retained
        .iter()
        .enumerate()
        .filter_map(|(wire, keep)| keep.then(|| one_hot(wire)))
        .collect();
    let targets: BTreeSet<_> = certificate
        .projected_rows
        .iter()
        .filter(|value| **value != BigUint::from(0u8) && !inputs.contains(*value))
        .cloned()
        .collect();
    if targets.len() != certificate.target_count
        || targets.into_iter().collect::<Vec<_>>() != certificate.targets
    {
        return false;
    }

    let h_basis = low_basis(certificate.h_basis.iter().cloned());
    if h_basis.len() != certificate.h_dimension {
        return false;
    }
    for vector in &certificate.h_basis {
        if (0..n).any(|wire| !certificate.retained[wire] && vector.bit(wire as u64)) {
            return false;
        }
    }
    let in_h = |value: &BigUint| low_remainder(value.clone(), &h_basis) == BigUint::from(0u8);

    let mut pool = inputs.clone();
    pool.extend(certificate.targets.iter().cloned());
    let outside: Vec<_> = pool.into_iter().filter(|value| !in_h(value)).collect();
    let mut w0 = Vec::new();
    w0.extend(inputs.iter().filter(|value| in_h(value)).cloned());
    for left in 0..outside.len() {
        for right in left + 1..outside.len() {
            let difference = &outside[left] ^ &outside[right];
            if in_h(&difference) {
                w0.push(difference);
            }
        }
    }
    let w0_basis = low_basis(w0);
    let mut extended: Vec<_> = w0_basis.values().cloned().collect();
    extended.extend(
        certificate
            .targets
            .iter()
            .filter(|value| in_h(value))
            .cloned(),
    );
    let extended_basis = low_basis(extended);
    let obstruction = extended_basis.len() - w0_basis.len();
    if obstruction != certificate.obstruction_rank || w0_basis.len() != certificate.w0_dimension {
        return false;
    }
    certificate.lower_bound == certificate.deletions + certificate.target_count + obstruction
}

fn cheap_lower_bound(rows: &[BigUint]) -> usize {
    let n = rows.len();
    let mut defects = Vec::with_capacity(n);
    let mut changed_rows = 0usize;
    let mut changed_columns = vec![false; n];
    for (row, value) in rows.iter().enumerate() {
        let defect = value ^ one_hot(row);
        if defect != BigUint::from(0u8) {
            changed_rows += 1;
        }
        for (column, changed) in changed_columns.iter_mut().enumerate() {
            *changed |= defect.bit(column as u64);
        }
        defects.push(defect);
    }
    let rank_defect = high_basis(defects).len();
    changed_rows
        .max(
            changed_columns
                .into_iter()
                .filter(|changed| *changed)
                .count(),
        )
        .max(rank_defect)
}

fn evaluate(
    state: &MatrixState,
    max_n: u32,
    max_checker_pairs: usize,
) -> Result<Stage26Values, EvaluationError> {
    let n = state.dimension();
    let expected_alphabet = format!("all_to_all_unit_cnot:{n}");
    if state.alphabet_id() != expected_alphabet.as_str() {
        return Err(EvaluationError::UnsupportedAlphabet);
    }
    let pool_size = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .ok_or(EvaluationError::ResourceLimit(
            "stage26 checker pool size overflow",
        ))?;
    let checker_pairs = pool_size
        .checked_mul(pool_size.saturating_sub(1))
        .map(|pairs| pairs / 2)
        .ok_or(EvaluationError::ResourceLimit(
            "stage26 checker pair count overflow",
        ))?;
    if n > max_n || checker_pairs > max_checker_pairs {
        return Err(EvaluationError::ResourceLimit(
            "stage26 source/checker budget exceeded",
        ));
    }

    let certificate = make_certificate(state);
    if !independently_verify(state, &certificate) {
        return Err(EvaluationError::VerificationFailed(
            "independent stage26 certificate checker rejected the generated bound",
        ));
    }
    let n = n as usize;
    let mut cosets = Vec::with_capacity(n);
    let mut multiplicities = HashMap::<BigUint, usize>::new();
    for value in &certificate.projected_rows {
        let coset = high_remainder(value.clone(), &certificate.h_basis);
        cosets.push(coset.clone());
        *multiplicities.entry(value.clone()).or_default() += 1;
    }
    let mut f04 = Vec::with_capacity(n * n * 3);
    for row in 0..n {
        for column in 0..n {
            let coset = &cosets[row];
            f04.push(
                if coset != &BigUint::from(0u8) && coset == &cosets[column] {
                    1.0
                } else {
                    0.0
                },
            );
            f04.push(
                if certificate.projected_rows[row] == certificate.projected_rows[column] {
                    1.0
                } else {
                    0.0
                },
            );
            f04.push(multiplicities[&certificate.projected_rows[row]] as f32);
        }
    }
    let cheap = cheap_lower_bound(&certificate.rows);
    let f03 = vec![
        certificate.deletions as f32,
        certificate.target_count as f32,
        certificate.obstruction_rank as f32,
        certificate.h_dimension as f32,
        certificate.w0_dimension as f32,
    ];
    let f05 = vec![
        certificate.lower_bound as f32,
        certificate.lower_bound as f32 - cheap as f32,
    ];
    Ok(Stage26Values { f03, f04, f05 })
}

struct SharedStage26 {
    max_n: u32,
    max_checker_pairs: usize,
    cache: Mutex<VecDeque<(StateIdentity, Arc<Stage26Values>)>>,
}

impl SharedStage26 {
    fn values(&self, state: &MatrixState) -> Result<Arc<Stage26Values>, EvaluationError> {
        let identity = StateIdentity::from(state);
        if let Ok(cache) = self.cache.lock() {
            if let Some((_, values)) = cache.iter().find(|(key, _)| *key == identity) {
                return Ok(Arc::clone(values));
            }
        } else {
            return Err(EvaluationError::VerificationFailed(
                "stage26 shared cache mutex poisoned",
            ));
        }
        let values = Arc::new(evaluate(state, self.max_n, self.max_checker_pairs)?);
        let mut cache = self.cache.lock().map_err(|_| {
            EvaluationError::VerificationFailed("stage26 shared cache mutex poisoned")
        })?;
        if let Some(position) = cache.iter().position(|(key, _)| *key == identity) {
            cache.remove(position);
        }
        cache.push_front((identity, Arc::clone(&values)));
        while cache.len() > CACHE_CAPACITY {
            cache.pop_back();
        }
        Ok(values)
    }
}

struct Stage26Producer {
    spec: FeatureSpec,
    shared: Arc<SharedStage26>,
}

impl FeatureProducer for Stage26Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let values = match self.shared.values(state) {
            Ok(values) => values,
            Err(EvaluationError::UnsupportedAlphabet) => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "stage26_requires_full_unit_alphabet",
                );
            }
            Err(EvaluationError::ResourceLimit(reason)) => {
                return unavailable(FeatureStatus::ResourceLimit, reason);
            }
            Err(EvaluationError::VerificationFailed(reason)) => {
                panic!("{reason}");
            }
        };
        let (shape, payload) = match self.spec.id.ordinal() {
            3 => (vec![5], values.f03.clone()),
            4 => {
                let n = state.dimension() as usize;
                (vec![n, n, 3], values.f04.clone())
            }
            5 => (vec![2], values.f05.clone()),
            _ => unreachable!("stage26 registered only F03-F05"),
        };
        FeatureComputation::Computed(FeatureValue {
            shape,
            values: payload,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

/// Registers exact F03/F04/F05 stage-26 features using the source defaults.
pub fn register_stage26_features(registry: &mut FeatureRegistry) -> Result<(), FeatureBuildError> {
    register_stage26_features_with_budget(registry, DEFAULT_MAX_N, DEFAULT_MAX_CHECKER_PAIRS)
}

/// Registers F03/F04/F05 with explicit width and independent-checker budgets.
pub fn register_stage26_features_with_budget(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_checker_pairs: usize,
) -> Result<(), FeatureBuildError> {
    if max_n == 0 || max_checker_pairs == 0 {
        return Err(FeatureBuildError(
            "stage26 requires positive width and checker-pair budgets".into(),
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"stage26_checked_features_v1");
    digest.update(STAGE26_CERTIFICATE_SOURCE_SHA256.as_bytes());
    digest.update(STAGE26_CHECKER_SOURCE_SHA256.as_bytes());
    digest.update(STAGE26_PROOF_NOTES_SHA256.as_bytes());
    digest.update(max_n.to_le_bytes());
    digest.update((max_checker_pairs as u64).to_le_bytes());
    let version = format!(
        "stage26_checked_features_v1:{}",
        hex_digest(digest.finalize().as_slice())
    );
    let shared = Arc::new(SharedStage26 {
        max_n,
        max_checker_pairs,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    });
    for (family, ordinal, channels) in STAGE26_IDS {
        let id = FeatureId::new(family, ordinal).expect("stage26 IDs are in the manifest");
        let role = if ordinal == 4 {
            FeatureRole::Pair
        } else {
            FeatureRole::Global
        };
        let spec = feature_spec(id).with_provider_metadata(
            "checked_stage26_canonical_projection;global_lower_bound_not_optimum",
            &version,
            ["M", "stage26_source", "independent_certificate_checker"],
            "exact",
            "T2",
            std::iter::empty::<&str>(),
            false,
        );
        let spec = FeatureSpec {
            role,
            channels,
            ..spec
        };
        registry.register(Box::new(Stage26Producer {
            spec,
            shared: Arc::clone(&shared),
        }))?;
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
