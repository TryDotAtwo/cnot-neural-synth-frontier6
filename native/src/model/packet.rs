//! Checked conversion from packet metadata and packed states to model tensors.

use super::ModelError;
use crate::packet::FeaturePacket;

pub(super) fn shape_i64(shape: &[usize]) -> Result<Vec<i64>, ModelError> {
    shape
        .iter()
        .map(|extent| {
            i64::try_from(*extent)
                .map_err(|_| ModelError("feature shape exceeds tensor index range".into()))
        })
        .collect()
}

pub(super) fn matrix_batch_values(packet: &FeaturePacket) -> Result<Vec<f32>, ModelError> {
    let total = packet
        .batch
        .checked_mul(packet.nodes)
        .and_then(|value| value.checked_mul(packet.nodes))
        .ok_or_else(|| ModelError("matrix batch size overflow".into()))?;
    let mut values = vec![0.0f32; total];
    for (batch_index, state) in packet.state_keys.iter().enumerate() {
        let dimension = usize::try_from(state.dimension)
            .map_err(|_| ModelError("matrix dimension exceeds host index range".into()))?;
        let words_per_row = dimension
            .checked_add(63)
            .ok_or_else(|| ModelError("matrix row width overflow".into()))?
            / 64;
        for row in 0..dimension {
            for column in 0..dimension {
                let word = state.packed_rows[row * words_per_row + column / 64];
                if (word >> (column % 64)) & 1 != 0 {
                    values[(batch_index * packet.nodes + row) * packet.nodes + column] = 1.0;
                }
            }
        }
    }
    Ok(values)
}
