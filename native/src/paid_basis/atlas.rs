//! Exhaustive breadth-first atlas for exact local paid-basis optimization.

use super::{BasisSide, ExactBasisOptimization, PaidBasisError};
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;

#[derive(Debug)]
struct ExactAtlas {
    distance: Vec<u8>,
    parent: Vec<Option<usize>>,
    parent_gate: Vec<Option<(usize, usize)>>,
    population: Vec<u8>,
}

struct AtlasCache {
    source: [OnceLock<ExactAtlas>; 5],
    target: [OnceLock<ExactAtlas>; 5],
}

static ATLAS_CACHE: OnceLock<AtlasCache> = OnceLock::new();

/// Returns the exact minimum paid price over every local GL(width, 2) basis.
///
/// `vectors` stores one binary vector per row. For `Source`, a gate `(c,t)`
/// changes each row by `row[c] ^= row[t]`; for `Target` it changes
/// `row[t] ^= row[c]`. The gate order and equal-cost tie break match the
/// source `code/exact_basis.py` atlas construction.
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

fn atlas(width: usize, side: BasisSide) -> &'static ExactAtlas {
    let cache = ATLAS_CACHE.get_or_init(|| AtlasCache {
        source: std::array::from_fn(|_| OnceLock::new()),
        target: std::array::from_fn(|_| OnceLock::new()),
    });
    let slots = match side {
        BasisSide::Source => &cache.source,
        BasisSide::Target => &cache.target,
    };
    slots[width].get_or_init(|| ExactAtlas::build(width, side))
}

impl ExactAtlas {
    fn build(width: usize, side: BasisSide) -> Self {
        debug_assert!((1..=4).contains(&width));
        let state_count = 1usize << width;
        let identity = (0..state_count).collect::<Vec<_>>();
        let identity_key = encode_mapping(&identity, width);
        let mut states = vec![identity_key];
        let mut distance = vec![0u8];
        let mut parent = vec![None];
        let mut parent_gate = vec![None];
        let mut seen = HashMap::new();
        seen.insert(identity_key, 0usize);
        let mut queue = VecDeque::from([0usize]);

        while let Some(index) = queue.pop_front() {
            let state = states[index];
            for control in 0..width {
                for target in 0..width {
                    if control == target {
                        continue;
                    }
                    let next = apply_to_mapping(state, width, side, control, target);
                    if seen.contains_key(&next) {
                        continue;
                    }
                    let next_index = states.len();
                    seen.insert(next, next_index);
                    states.push(next);
                    distance.push(distance[index] + 1);
                    parent.push(Some(index));
                    parent_gate.push(Some((control, target)));
                    queue.push_back(next_index);
                }
            }
        }

        let mut population = Vec::with_capacity(states.len() * state_count);
        for state in &states {
            for code in 0..state_count {
                population.push(state_image(*state, width, code).count_ones() as u8);
            }
        }

        Self {
            distance,
            parent,
            parent_gate,
            population,
        }
    }
}

fn encode_mapping(mapping: &[usize], width: usize) -> u64 {
    mapping
        .iter()
        .enumerate()
        .fold(0u64, |packed, (input, output)| {
            packed | ((*output as u64) << (input * width))
        })
}

fn state_image(state: u64, width: usize, code: usize) -> usize {
    let mask = (1u64 << width) - 1;
    ((state >> (code * width)) & mask) as usize
}

fn apply_to_mapping(
    state: u64,
    width: usize,
    side: BasisSide,
    control: usize,
    target: usize,
) -> u64 {
    let mut packed = 0u64;
    for input in 0..(1usize << width) {
        let mut output = state_image(state, width, input);
        match side {
            BasisSide::Source => output ^= ((output >> target) & 1) << control,
            BasisSide::Target => output ^= ((output >> control) & 1) << target,
        }
        packed |= (output as u64) << (input * width);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::{BasisSide, atlas};

    #[test]
    fn atlas_enumerates_every_local_linear_basis() {
        let expected = [0usize, 1, 6, 168, 20_160];
        for width in 1..=4 {
            assert_eq!(
                atlas(width, BasisSide::Source).population.len() / (1usize << width),
                expected[width]
            );
            assert_eq!(
                atlas(width, BasisSide::Target).population.len() / (1usize << width),
                expected[width]
            );
        }
    }
}
