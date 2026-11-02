//! H06 exact pair differences across physical matrix views.
//!
//! The provider follows research_v3.features.symmetry_alignment: for each
//! ordered physical wire pair it concatenates the differences in A15, A16,
//! and D03 descriptors for M^T, M^-1, and M^-T relative to M.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::Mutex;

const H06_CHANNELS: usize = 15;
const VIEW_COUNT: usize = 4;
const DESCRIPTOR_CHANNELS: usize = 5;
const CACHE_CAPACITY: usize = 32;

#[derive(Clone)]
struct CacheEntry {
    state: StateIdentity,
    max_bytes: Option<usize>,
    value: FeatureComputation,
}

struct H06Producer {
    spec: FeatureSpec,
    cache: Mutex<VecDeque<CacheEntry>>,
}

impl FeatureProducer for H06Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        let identity = StateIdentity::from(state);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache
                .iter()
                .position(|entry| entry.state == identity && entry.max_bytes == policy.max_bytes)
            {
                if let Some(entry) = cache.remove(index) {
                    let result = entry.value.clone();
                    cache.push_front(entry);
                    return result;
                }
            }
        }

        let value = compute_h06(state, policy.max_bytes);
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CacheEntry {
                state: identity,
                max_bytes: policy.max_bytes,
                value: value.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        value
    }
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn compute_h06(state: &MatrixState, max_bytes: Option<usize>) -> FeatureComputation {
    let n = state.dimension() as usize;
    let cells = match n.checked_mul(n) {
        Some(value) => value,
        None => return unavailable(FeatureStatus::ResourceLimit, "h06_matrix_size_overflow"),
    };
    let elements = match cells.checked_mul(H06_CHANNELS) {
        Some(value) => value,
        None => return unavailable(FeatureStatus::ResourceLimit, "h06_output_size_overflow"),
    };
    let bytes = match elements.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "h06_output_byte_size_overflow",
            )
        }
    };
    if max_bytes.is_some_and(|budget| bytes > budget) {
        return unavailable(
            FeatureStatus::ResourceLimit,
            "h06_output_exceeds_policy_byte_budget",
        );
    }

    let views = match matrix_views(state) {
        Ok(views) => views,
        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
    };
    let base = match pair_descriptors(&views[0], n) {
        Ok(value) => value,
        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
    };
    let mut output = Vec::new();
    if output.try_reserve_exact(elements).is_err() {
        return unavailable(FeatureStatus::ResourceLimit, "h06_output_allocation_failed");
    }
    output.resize(elements, 0.0f32);

    for (view_index, view) in views.iter().enumerate().skip(1) {
        let descriptors = match pair_descriptors(view, n) {
            Ok(value) => value,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        for pair in 0..cells {
            for channel in 0..DESCRIPTOR_CHANNELS {
                let difference = descriptors[pair][channel] - base[pair][channel];
                let target = pair * H06_CHANNELS + (view_index - 1) * DESCRIPTOR_CHANNELS + channel;
                output[target] = difference as f32;
            }
        }
    }

    FeatureComputation::Computed(FeatureValue {
        shape: vec![n, n, H06_CHANNELS],
        values: output,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}

fn matrix_views(state: &MatrixState) -> Result<[Vec<u64>; VIEW_COUNT], String> {
    let n = state.dimension() as usize;
    let words = state.words_per_row();
    let rows = state.packed_rows();
    let transpose = transpose_packed(rows, n, words)?;
    let inverse = crate::structural::gf2_inverse(rows, n)?;
    let inverse_transpose = transpose_packed(&inverse, n, words)?;
    Ok([rows.to_vec(), transpose, inverse, inverse_transpose])
}

fn transpose_packed(rows: &[u64], n: usize, words: usize) -> Result<Vec<u64>, String> {
    let length = n
        .checked_mul(words)
        .ok_or_else(|| "h06_transpose_size_overflow".to_owned())?;
    let mut transposed = Vec::new();
    transposed
        .try_reserve_exact(length)
        .map_err(|_| "h06_transpose_allocation_failed".to_owned())?;
    transposed.resize(length, 0u64);
    for row in 0..n {
        for source_word in 0..words {
            let mut remaining = rows[row * words + source_word];
            while remaining != 0 {
                let column = source_word * 64 + remaining.trailing_zeros() as usize;
                if column < n {
                    transposed[column * words + row / 64] |= 1u64 << (row % 64);
                }
                remaining &= remaining - 1;
            }
        }
    }
    Ok(transposed)
}

fn pair_descriptors(rows: &[u64], n: usize) -> Result<Vec<[i64; DESCRIPTOR_CHANNELS]>, String> {
    let words = n
        .checked_add(63)
        .ok_or_else(|| "h06_word_count_overflow".to_owned())?
        / 64;
    let cells = n
        .checked_mul(n)
        .ok_or_else(|| "h06_descriptor_size_overflow".to_owned())?;
    let expected_rows = n
        .checked_mul(words)
        .ok_or_else(|| "h06_packed_matrix_size_overflow".to_owned())?;
    if rows.len() != expected_rows {
        return Err("h06_matrix_view_has_an_invalid_packed_length".into());
    }

    let mut columns = Vec::new();
    let column_words = words;
    let column_length = n
        .checked_mul(column_words)
        .ok_or_else(|| "h06_column_view_size_overflow".to_owned())?;
    columns
        .try_reserve_exact(column_length)
        .map_err(|_| "h06_column_view_allocation_failed".to_owned())?;
    columns.resize(column_length, 0u64);
    for row in 0..n {
        for word_index in 0..words {
            let word = rows[row * words + word_index];
            let mut remaining = word;
            while remaining != 0 {
                let column = word_index * 64 + remaining.trailing_zeros() as usize;
                if column < n {
                    columns[column * column_words + row / 64] |= 1u64 << (row % 64);
                }
                remaining &= remaining - 1;
            }
        }
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(cells)
        .map_err(|_| "h06_descriptor_allocation_failed".to_owned())?;
    for left in 0..n {
        for right in 0..n {
            let mut row_distance = 0u64;
            let mut column_distance = 0u64;
            let mut one_zero = 0u64;
            let mut zero_one = 0u64;
            let mut one_one = 0u64;
            for word in 0..words {
                let a = rows[left * words + word];
                let b = rows[right * words + word];
                row_distance += u64::from((a ^ b).count_ones());
                one_zero += u64::from((a & !b).count_ones());
                zero_one += u64::from((b & !a).count_ones());
                one_one += u64::from((a & b).count_ones());

                let column_a = columns[left * column_words + word];
                let column_b = columns[right * column_words + word];
                column_distance += u64::from((column_a ^ column_b).count_ones());
            }
            let a15 = i64::try_from(row_distance)
                .map_err(|_| "h06_row_distance_exceeds_i64".to_owned())?;
            let a16 = i64::try_from(column_distance)
                .map_err(|_| "h06_column_distance_exceeds_i64".to_owned())?;
            let d03_10 =
                i64::try_from(one_zero).map_err(|_| "h06_d03_count_exceeds_i64".to_owned())?;
            let d03_01 =
                i64::try_from(zero_one).map_err(|_| "h06_d03_count_exceeds_i64".to_owned())?;
            let d03_11 =
                i64::try_from(one_one).map_err(|_| "h06_d03_count_exceeds_i64".to_owned())?;
            output.push([a15, a16, d03_10, d03_01, d03_11]);
        }
    }
    Ok(output)
}

/// Register H06's exact 15-channel physical pair descriptor.
pub fn register_symmetry_alignment_h06(
    registry: &mut FeatureRegistry,
) -> Result<(), FeatureBuildError> {
    let id = FeatureId::new(b'H', 6).expect("H06 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Pair;
    spec.channels = H06_CHANNELS;
    let spec = spec.with_provider_metadata(
        "physical_pair_differences_A15_A16_width2_D03_across_T_inverse_inverseT",
        "physical_symmetry_alignment_v1",
        ["M", "exact_GF2_inverse", "physical_wire_matching"],
        "exact",
        "T1",
        std::iter::empty::<&str>(),
        false,
    );
    registry.register(Box::new(H06Producer {
        spec,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}
