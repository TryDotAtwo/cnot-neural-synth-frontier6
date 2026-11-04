mod accumulate;

use self::accumulate::MomentAccumulator;
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
        let words_per_row = state.words_per_row() as usize;
        let column_words = n.checked_add(63)? / 64;
        let lambda = n as f64 * config.mc_tau;
        if !lambda.is_finite() {
            return None;
        }

        let mut accumulator = MomentAccumulator::new(n)?;
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
            accumulator.observe_columns(&columns);
        }

        Some(accumulator.finish(config.mc_samples, total_gates))
    }
}
