use super::{unavailable, DESCRIPTOR_CHANNELS, H06_CHANNELS, VIEW_COUNT};
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::FeatureStatus;
use crate::state::MatrixState;

pub(super) fn compute_h06(state: &MatrixState, max_bytes: Option<usize>) -> FeatureComputation {
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
