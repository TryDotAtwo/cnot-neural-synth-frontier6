//! Coordinate restriction, relation extraction, and feature buffer assembly.

use super::super::context::{bit, nullspace, set_bit, words_for, SharedRelationContext};
use crate::features::FeatureValue;
use crate::state::MatrixState;

pub(super) fn compute_tokens(
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
