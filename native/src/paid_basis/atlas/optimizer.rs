//! Cost minimization and independent candidate verification over an atlas.

use super::super::{BasisSide, ExactBasisOptimization, PaidBasisError};
use super::atlas;

pub fn optimize_exact_basis(
    vectors: &[Vec<u8>],
    side: BasisSide,
) -> Result<ExactBasisOptimization, PaidBasisError> {
    let width = vectors
        .first()
        .map(Vec::len)
        .ok_or_else(|| PaidBasisError("basis block must contain at least one vector".into()))?;
    if !(1..=4).contains(&width) {
        return Err(PaidBasisError(
            "exact paid-basis atlas supports local width 1..=4".into(),
        ));
    }
    if vectors
        .iter()
        .any(|row| row.len() != width || row.iter().any(|bit| *bit > 1))
    {
        return Err(PaidBasisError(
            "basis vectors must have a consistent width and contain only bits".into(),
        ));
    }

    let mut direct_cost = 0usize;
    let mut histogram = vec![0usize; 1usize << width];
    for row in vectors {
        let mut code = 0usize;
        for (bit_index, bit) in row.iter().copied().enumerate() {
            direct_cost = direct_cost
                .checked_add(usize::from(bit))
                .ok_or_else(|| PaidBasisError("direct basis cost overflow".into()))?;
            code |= usize::from(bit) << bit_index;
        }
        histogram[code] = histogram[code]
            .checked_add(1)
            .ok_or_else(|| PaidBasisError("basis histogram count overflow".into()))?;
    }

    let atlas = atlas(width, side);
    let mut best_index = 0usize;
    let mut best_cost = usize::MAX;
    let code_count = histogram.len();
    for (state_index, population) in atlas.population.chunks_exact(code_count).enumerate() {
        let mut transformed_core_cost = 0usize;
        for (count, transformed_weight) in histogram.iter().copied().zip(population.iter().copied())
        {
            if count == 0 {
                continue;
            }
            let term = usize::from(transformed_weight)
                .checked_mul(count)
                .ok_or_else(|| PaidBasisError("transformed core cost overflow".into()))?;
            transformed_core_cost = transformed_core_cost
                .checked_add(term)
                .ok_or_else(|| PaidBasisError("transformed core cost overflow".into()))?;
        }
        let preparation_cost = usize::from(atlas.distance[state_index])
            .checked_mul(2)
            .ok_or_else(|| PaidBasisError("paid preparation cost overflow".into()))?;
        let paid_cost = preparation_cost
            .checked_add(transformed_core_cost)
            .ok_or_else(|| PaidBasisError("paid basis cost overflow".into()))?;
        // The source uses first-occurrence argmin. Iterating atlas BFS order and
        // replacing only on strict improvement preserves that tie break.
        if paid_cost < best_cost {
            best_cost = paid_cost;
            best_index = state_index;
        }
    }

    let mut operations = Vec::new();
    let mut cursor = best_index;
    while let Some(parent) = atlas.parent[cursor] {
        operations.push(atlas.parent_gate[cursor].ok_or_else(|| {
            PaidBasisError("exact atlas parent is missing its generating gate".into())
        })?);
        cursor = parent;
    }
    operations.reverse();

    let mut transformed_vectors = vectors.to_vec();
    for &(control, target) in &operations {
        for row in &mut transformed_vectors {
            match side {
                BasisSide::Source => row[control] ^= row[target],
                BasisSide::Target => row[target] ^= row[control],
            }
        }
    }
    let independently_counted = operations
        .len()
        .checked_mul(2)
        .and_then(|preparation| {
            transformed_vectors
                .iter()
                .flatten()
                .try_fold(preparation, |sum, bit| sum.checked_add(usize::from(*bit)))
        })
        .ok_or_else(|| PaidBasisError("verified paid basis cost overflow".into()))?;
    if independently_counted != best_cost || best_cost > direct_cost {
        return Err(PaidBasisError(
            "exact atlas candidate failed its paid-cost invariant".into(),
        ));
    }

    Ok(ExactBasisOptimization {
        operations,
        direct_cost,
        paid_cost: best_cost,
        transformed_vectors,
    })
}
