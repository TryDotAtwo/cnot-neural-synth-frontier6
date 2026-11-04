use super::super::{DiffusionConfig, MAX_MC_CACHE_ENTRIES};
use super::ProfileCache;
use crate::numpy_rng::NumpyRng;
use crate::state::MatrixState;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct McKey {
    dimension: u32,
    rows: Vec<u64>,
    seed: u64,
}

pub(in crate::diffusion) struct JointMoments {
    pub(in crate::diffusion) samples: usize,
    pub(in crate::diffusion) total_gates: u64,
    pub(in crate::diffusion) mean_weight: Vec<f64>,
    pub(in crate::diffusion) covariance_weight: Vec<f64>,
    pub(in crate::diffusion) covariance_weight_intersection: Vec<f64>,
    pub(in crate::diffusion) covariance_weight_xor: Vec<f64>,
    pub(in crate::diffusion) mean_intersection: Vec<f64>,
    pub(in crate::diffusion) mean_xor: Vec<f64>,
    pub(in crate::diffusion) mean_character_pair: Vec<f64>,
    pub(in crate::diffusion) covariance_character_pair: Vec<f64>,
    pub(in crate::diffusion) standard_error_intersection: Vec<f64>,
    pub(in crate::diffusion) standard_error_xor: Vec<f64>,
    pub(in crate::diffusion) standard_error_character_pair: Vec<f64>,
}

impl ProfileCache {
    pub(in crate::diffusion) fn get_mc(
        &self,
        state: &MatrixState,
        seed: u64,
    ) -> Option<Arc<JointMoments>> {
        let key = McKey {
            dimension: state.dimension(),
            rows: state.packed_rows().to_vec(),
            seed,
        };
        {
            let profiles = self
                .mc_profiles
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(profile) = profiles.get(&key) {
                return Some(Arc::clone(profile));
            }
        }

        let profile = Arc::new(JointMoments::sample(state, self.config, seed)?);
        let mut profiles = self
            .mc_profiles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = profiles.get(&key) {
            return Some(Arc::clone(existing));
        }
        if profiles.len() >= MAX_MC_CACHE_ENTRIES {
            profiles.clear();
        }
        profiles.insert(key, Arc::clone(&profile));
        Some(profile)
    }
}

impl JointMoments {
    fn sample(state: &MatrixState, config: DiffusionConfig, seed: u64) -> Option<Self> {
        let n = state.dimension() as usize;
        let pair_count = n.checked_mul(n)?;
        let words_per_row = state.words_per_row() as usize;
        let column_words = n.checked_add(63)? / 64;
        let lambda = n as f64 * config.mc_tau;
        if !lambda.is_finite() {
            return None;
        }

        let mut sum_weight = vec![0.0; n];
        let mut sum_weight_pair = vec![0.0; pair_count];
        let mut sum_weight_intersection = vec![0.0; pair_count];
        let mut sum_weight_xor = vec![0.0; pair_count];
        let mut sum_intersection = vec![0.0; pair_count];
        let mut sum_intersection_squared = vec![0.0; pair_count];
        let mut sum_xor = vec![0.0; pair_count];
        let mut sum_xor_squared = vec![0.0; pair_count];
        let mut sum_character = vec![0.0; n];
        let mut sum_character_pair = vec![0.0; pair_count];
        let mut sum_character_pair_squared = vec![0.0; pair_count];
        let mut total_gates = 0u64;
        let mut rng = NumpyRng::new(seed);

        for _ in 0..config.mc_samples {
            let mut rows = state
                .packed_rows()
                .chunks_exact(words_per_row)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let steps = if n > 1 { rng.poisson(lambda)? } else { 0 };
            total_gates = total_gates.checked_add(steps)?;
            for _ in 0..steps {
                let control = rng.integer(n)?;
                let mut target = rng.integer(n - 1)?;
                target += usize::from(target >= control);
                for word in 0..words_per_row {
                    rows[target][word] ^= rows[control][word];
                }
            }

            let mut columns = vec![vec![0u64; column_words]; n];
            for (row_index, row) in rows.iter().enumerate() {
                for column in 0..n {
                    if ((row[column / 64] >> (column % 64)) & 1) != 0 {
                        columns[column][row_index / 64] |= 1u64 << (row_index % 64);
                    }
                }
            }
            let weights = columns
                .iter()
                .map(|column| column.iter().map(|word| word.count_ones() as f64).sum())
                .collect::<Vec<f64>>();
            let characters = weights
                .iter()
                .map(|weight| {
                    if (*weight as usize) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect::<Vec<f64>>();
            for first in 0..n {
                sum_weight[first] += weights[first];
                sum_character[first] += characters[first];
                for second in 0..n {
                    let index = first * n + second;
                    sum_weight_pair[index] += weights[first] * weights[second];
                    let mut intersection = 0u32;
                    let mut xor_weight = 0u32;
                    for word in 0..column_words {
                        let left = columns[first][word];
                        let right = columns[second][word];
                        intersection += (left & right).count_ones();
                        xor_weight += (left ^ right).count_ones();
                    }
                    let intersection = f64::from(intersection);
                    let xor_weight = f64::from(xor_weight);
                    let character_pair = characters[first] * characters[second];
                    sum_intersection[index] += intersection;
                    sum_intersection_squared[index] += intersection * intersection;
                    sum_xor[index] += xor_weight;
                    sum_xor_squared[index] += xor_weight * xor_weight;
                    sum_weight_intersection[index] += weights[first] * intersection;
                    sum_weight_xor[index] += weights[first] * xor_weight;
                    sum_character_pair[index] += character_pair;
                    sum_character_pair_squared[index] += character_pair * character_pair;
                }
            }
        }

        let samples = config.mc_samples;
        let denominator = (samples - 1) as f64;
        let mean_weight = sum_weight
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_intersection = sum_intersection
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_xor = sum_xor
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_character = sum_character
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_character_pair = sum_character_pair
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();

        let mut covariance_weight = vec![0.0; pair_count];
        let mut covariance_weight_intersection = vec![0.0; pair_count];
        let mut covariance_weight_xor = vec![0.0; pair_count];
        let mut covariance_character_pair = vec![0.0; pair_count];
        let mut standard_error_intersection = vec![0.0; pair_count];
        let mut standard_error_xor = vec![0.0; pair_count];
        let mut standard_error_character_pair = vec![0.0; pair_count];
        for first in 0..n {
            for second in 0..n {
                let index = first * n + second;
                covariance_weight[index] = (sum_weight_pair[index]
                    - sum_weight[first] * mean_weight[second])
                    / denominator;
                covariance_weight_intersection[index] = (sum_weight_intersection[index]
                    - sum_weight[first] * mean_intersection[index])
                    / denominator;
                covariance_weight_xor[index] =
                    (sum_weight_xor[index] - sum_weight[first] * mean_xor[index]) / denominator;
                covariance_character_pair[index] =
                    mean_character_pair[index] - mean_character[first] * mean_character[second];
                standard_error_intersection[index] = standard_error(
                    sum_intersection[index],
                    sum_intersection_squared[index],
                    samples,
                );
                standard_error_xor[index] =
                    standard_error(sum_xor[index], sum_xor_squared[index], samples);
                standard_error_character_pair[index] = standard_error(
                    sum_character_pair[index],
                    sum_character_pair_squared[index],
                    samples,
                );
            }
        }

        Some(Self {
            samples,
            total_gates,
            mean_weight,
            covariance_weight,
            covariance_weight_intersection,
            covariance_weight_xor,
            mean_intersection,
            mean_xor,
            mean_character_pair,
            covariance_character_pair,
            standard_error_intersection,
            standard_error_xor,
            standard_error_character_pair,
        })
    }
}

fn standard_error(sum: f64, sum_squared: f64, samples: usize) -> f64 {
    let samples_f64 = samples as f64;
    let variance = ((sum_squared - sum * sum / samples_f64) / (samples - 1) as f64).max(0.0);
    variance.sqrt() / samples_f64.sqrt()
}
