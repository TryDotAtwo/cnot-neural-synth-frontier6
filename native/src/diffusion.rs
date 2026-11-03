//! Opt-in source-relative CNOT diffusion features G01-G04 and G06-G10.
//!
//! The source computes finite continuous-time Markov semigroups over exact
//! column-weight and rank-2 pair-histogram classes. Uniformization uses the same transition counts and
//! retains the source's floating-semigroup status; it does not claim a formal
//! numerical error certificate.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::numpy_rng::NumpyRng;
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod profiles;
use profiles::{pair_state_count, ColumnProfile, PairProfile};

const MAX_CACHED_WIDTHS: usize = 16;
const MAX_MC_CACHE_ENTRIES: usize = 8;
const MAX_MC_WORKING_BYTES: usize = 256 * 1024 * 1024;
pub(super) const MAX_UNIFORMIZATION_TERMS: usize = 1_000_000;
pub(super) const POISSON_TAIL_SIGMAS: f64 = 12.0;
pub(super) const POISSON_TAIL_MARGIN: f64 = 32.0;

#[derive(Clone, Copy, Debug)]
pub struct DiffusionConfig {
    /// Four values are fixed by the canonical 103-feature channel layout.
    pub taus: [f64; 4],
    pub floor: f64,
    pub max_n: u32,
    /// Explicit state-space limit for the rank-2 pair projection G06.
    pub max_pair_states: usize,
    /// Number of shared random CNOT words sampled by G07-G10.
    pub mc_samples: usize,
    /// Source time multiplier; the sampled Poisson time is `n * mc_tau`.
    pub mc_tau: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            taus: [0.25, 1.0, 4.0, 16.0],
            floor: 1e-30,
            max_n: 256,
            max_pair_states: 10_000,
            mc_samples: 256,
            mc_tau: 1.0,
        }
    }
}

/// Register source-relative diffusion producers G01-G04 and G06-G10.
///
/// Registration is opt-in, like the Python source. The provider identities bind
/// emitted values to their numeric parameters and explicit resource budgets.
pub fn register_diffusion(
    registry: &mut FeatureRegistry,
    config: DiffusionConfig,
) -> Result<(), FeatureBuildError> {
    validate_config(config)?;
    let ids = [
        FeatureId::new(b'G', 1).unwrap(),
        FeatureId::new(b'G', 2).unwrap(),
        FeatureId::new(b'G', 3).unwrap(),
        FeatureId::new(b'G', 4).unwrap(),
        FeatureId::new(b'G', 6).unwrap(),
    ];
    let mc_ids = [
        FeatureId::new(b'G', 7).unwrap(),
        FeatureId::new(b'G', 8).unwrap(),
        FeatureId::new(b'G', 9).unwrap(),
        FeatureId::new(b'G', 10).unwrap(),
    ];
    let specs = registry.specs();
    if let Some(id) = ids
        .iter()
        .chain(mc_ids.iter())
        .find(|id| specs[id.index()].producer_version != "unconfigured")
    {
        return Err(FeatureBuildError(format!(
            "{id} already has a configured provider"
        )));
    }

    let version = producer_version(config);
    let cache = Arc::new(ProfileCache {
        config,
        profiles: Mutex::new(HashMap::new()),
        pair_profiles: Mutex::new(HashMap::new()),
        mc_profiles: Mutex::new(HashMap::new()),
    });
    for id in ids {
        let mut spec = feature_spec(id);
        spec.scope = if id.ordinal() == 6 {
            "full_labeled_matrix;histogram_class_mass_not_labeled_endpoint".into()
        } else {
            "full_labeled_matrix;source_relative_uniform_CNOT".into()
        };
        spec.producer_version = version.clone();
        spec.dependencies = if id.ordinal() == 6 {
            vec!["M".into(), "source_relative_rank2_pair_chain".into()]
        } else {
            vec!["M".into(), "source_relative_column_chain".into()]
        };
        spec.exactness = "floating_semigroup".into();
        spec.cost_tier = if id.ordinal() == 6 { "T2" } else { "T1" }.into();
        spec.wire_roles.clear();
        spec.requires_solver_assisted = false;
        registry.register(Box::new(DiffusionProducer {
            spec,
            cache: Arc::clone(&cache),
        }))?;
    }
    let mc_version = mc_producer_version(config);
    for id in mc_ids {
        let mut spec = feature_spec(id);
        spec.scope =
            "full_labeled_matrix;shared_word_MC_joint_moments_empirical_uncertainty".into();
        spec.producer_version = mc_version.clone();
        spec.dependencies = vec!["M".into(), "shared_word_uniform_ordered_CNOT_MC".into()];
        spec.exactness = "monte_carlo".into();
        spec.cost_tier = "T2".into();
        spec.wire_roles.clear();
        spec.requires_solver_assisted = false;
        registry.register(Box::new(MonteCarloProducer {
            spec,
            cache: Arc::clone(&cache),
        }))?;
    }
    Ok(())
}

fn validate_config(config: DiffusionConfig) -> Result<(), FeatureBuildError> {
    if config.max_n == 0 {
        return Err(FeatureBuildError("diffusion max_n must be positive".into()));
    }
    if config.max_pair_states == 0 {
        return Err(FeatureBuildError(
            "diffusion max_pair_states must be positive".into(),
        ));
    }
    if config.mc_samples < 2 {
        return Err(FeatureBuildError(
            "diffusion mc_samples must be at least two".into(),
        ));
    }
    if !config.mc_tau.is_finite() || config.mc_tau < 0.0 {
        return Err(FeatureBuildError(
            "diffusion mc_tau must be finite and nonnegative".into(),
        ));
    }
    if config.taus.iter().any(|tau| !tau.is_finite() || *tau < 0.0) {
        return Err(FeatureBuildError(
            "diffusion times must be finite and nonnegative".into(),
        ));
    }
    if !config.floor.is_finite() || !(0.0..1.0).contains(&config.floor) {
        return Err(FeatureBuildError(
            "diffusion floor must be finite and strictly between zero and one".into(),
        ));
    }
    Ok(())
}

fn producer_version(config: DiffusionConfig) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/source_relative_diffusion_v2\0");
    for tau in config.taus {
        digest.update(tau.to_bits().to_le_bytes());
    }
    digest.update(config.floor.to_bits().to_le_bytes());
    digest.update(config.max_n.to_le_bytes());
    digest.update((config.max_pair_states as u64).to_le_bytes());
    let hex = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("diffusion_v2:{hex}")
}

fn mc_producer_version(config: DiffusionConfig) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/shared_word_diffusion_monte_carlo_v1\0");
    digest.update(config.max_n.to_le_bytes());
    digest.update((config.mc_samples as u64).to_le_bytes());
    digest.update(config.mc_tau.to_bits().to_le_bytes());
    digest.update(b"numpy_2.2.6_default_rng_pcg64\0");
    let hex = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("diffusion_mc_v1:{hex}")
}

struct DiffusionProducer {
    spec: FeatureSpec,
    cache: Arc<ProfileCache>,
}

impl FeatureProducer for DiffusionProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        if self.spec.id.ordinal() == 6 {
            return self.compute_pair(state, n);
        }
        if n > self.cache.config.max_n as usize {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "column diffusion max_n={} exceeded by n={n}",
                    self.cache.config.max_n
                ),
            );
        }
        let Some(profile) = self.cache.get(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion checked state/iteration budget exceeded".into(),
            );
        };
        let Some(value_count) = n.checked_mul(self.spec.channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "column diffusion output allocation failed".into(),
            );
        }
        values.resize(value_count, 0.0f32);

        let row_weights = row_weights(state);
        let column_weights = column_weights(state);
        for wire in 0..n {
            let diagonal_bit = matrix_bit(state, wire, wire) as usize;
            let weight = if self.spec.id.ordinal() == 2 {
                row_weights[wire]
            } else {
                column_weights[wire]
            };
            let Some(other_weight) = weight.checked_sub(diagonal_bit) else {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "diagonal bit exceeds selected row/column weight".into(),
                );
            };
            let Some(class) = profile.class_index(diagonal_bit, other_weight) else {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "selected row/column class is outside the diffusion chain".into(),
                );
            };
            let base = wire * self.spec.channels;
            match self.spec.id.ordinal() {
                1 | 2 => {
                    for time_index in 0..4 {
                        let sample = profile.sample(time_index, class);
                        let offset = base + time_index * 7;
                        values[offset] = sample.raw as f32;
                        values[offset + 1] = profile.log_class_size[class] as f32;
                        values[offset + 2] = sample.stationary as f32;
                        values[offset + 3] = (sample.log_mass - sample.stationary.ln()) as f32;
                        values[offset + 4] = f32::from(sample.exact_zero);
                        values[offset + 5] = f32::from(sample.underflow);
                        values[offset + 6] = f32::from(sample.raw < self.cache.config.floor);
                    }
                }
                3 => {
                    for time_index in 0..4 {
                        let sample = profile.sample(time_index, class);
                        let offset = base + time_index * 4;
                        values[offset] = sample.log_mass as f32;
                        values[offset + 1] = f32::from(sample.exact_zero);
                        values[offset + 2] = f32::from(sample.underflow);
                        values[offset + 3] = f32::from(sample.raw < self.cache.config.floor);
                    }
                }
                4 => {
                    for time_index in 0..3 {
                        let current = profile.raw(time_index + 1, class);
                        let previous = profile.raw(time_index, class);
                        values[base + time_index] = (current - previous) as f32;
                    }
                }
                _ => {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "diffusion producer received an unregistered feature ID".into(),
                    );
                }
            }
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, self.spec.channels],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

struct MonteCarloProducer {
    spec: FeatureSpec,
    cache: Arc<ProfileCache>,
}

impl FeatureProducer for MonteCarloProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        if n > self.cache.config.max_n as usize {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "diffusion max_n={} exceeded by n={n}",
                    self.cache.config.max_n
                ),
            );
        }
        let Some(pair_count) = n.checked_mul(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo pair count overflow".into(),
            );
        };
        let Some(working_bytes) = pair_count
            .checked_mul(self.cache.config.mc_samples)
            .and_then(|value| value.checked_mul(8 * 8))
        else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo working-set size overflow".into(),
            );
        };
        if working_bytes > MAX_MC_WORKING_BYTES {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "Monte Carlo working set {working_bytes} bytes exceeds {MAX_MC_WORKING_BYTES}"
                ),
            );
        }
        let Some(moments) = self.cache.get_mc(state, policy.seed) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo sampling exceeded a checked allocation or numeric budget".into(),
            );
        };

        let channels = self.spec.channels;
        let output_cells = if self.spec.id.ordinal() == 7 {
            n
        } else {
            pair_count
        };
        let Some(value_count) = output_cells.checked_mul(channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "Monte Carlo output allocation failed".into(),
            );
        }

        match self.spec.id.ordinal() {
            7 => values.extend(moments.mean_weight.iter().map(|value| *value as f32)),
            8 => {
                for index in 0..pair_count {
                    values.extend([
                        moments.covariance_weight[index] as f32,
                        moments.mean_intersection[index] as f32,
                        moments.mean_xor[index] as f32,
                        moments.covariance_weight_intersection[index] as f32,
                        moments.covariance_weight_xor[index] as f32,
                    ]);
                }
            }
            9 => {
                for index in 0..pair_count {
                    values.extend([
                        moments.mean_character_pair[index] as f32,
                        moments.covariance_character_pair[index] as f32,
                    ]);
                }
            }
            10 => {
                let stationary_off_diagonal = -1.0 / (2.0f64.powi(n as i32) - 1.0);
                let gates_per_sample = moments.total_gates as f64 / moments.samples as f64;
                for first in 0..n {
                    for second in 0..n {
                        let index = first * n + second;
                        let stationary = if first == second {
                            1.0
                        } else {
                            stationary_off_diagonal
                        };
                        values.extend([
                            moments.standard_error_intersection[index] as f32,
                            moments.standard_error_xor[index] as f32,
                            moments.standard_error_character_pair[index] as f32,
                            (moments.mean_character_pair[index] - stationary).abs() as f32,
                            moments.samples as f32,
                            gates_per_sample as f32,
                            self.cache.config.mc_tau as f32,
                        ]);
                    }
                }
            }
            _ => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "Monte Carlo producer received an unregistered feature ID".into(),
                );
            }
        }
        if values.len() != value_count || values.iter().any(|value| !value.is_finite()) {
            return unavailable(
                FeatureStatus::Unsupported,
                "Monte Carlo producer generated a non-finite or malformed output".into(),
            );
        }

        let shape = if self.spec.id.ordinal() == 7 {
            vec![n, channels]
        } else {
            vec![n, n, channels]
        };
        FeatureComputation::Computed(FeatureValue {
            shape,
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl DiffusionProducer {
    fn compute_pair(&self, state: &MatrixState, n: usize) -> FeatureComputation {
        if n < 2 {
            return unavailable(
                FeatureStatus::Unsupported,
                "rank-2 pair projection requires n>=2".into(),
            );
        }
        let Some(state_count) = pair_state_count(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion state count overflow".into(),
            );
        };
        if state_count > self.cache.config.max_pair_states {
            return unavailable(
                FeatureStatus::ResourceLimit,
                format!(
                    "pair diffusion resource_limit: {state_count} states exceeds {}",
                    self.cache.config.max_pair_states
                ),
            );
        }
        let Some(profile) = self.cache.get_pair(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion checked allocation/iteration budget exceeded".into(),
            );
        };
        let Some(pair_count) = n.checked_mul(n) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output size overflow".into(),
            );
        };
        let Some(value_count) = pair_count.checked_mul(self.spec.channels) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output size overflow".into(),
            );
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(value_count).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "pair diffusion output allocation failed".into(),
            );
        }
        values.resize(value_count, 0.0f32);

        for first in 0..n {
            for second in 0..n {
                if first == second {
                    continue;
                }
                let histogram = pair_histogram(state, first, second);
                let Some(class) = profile.class_index(&histogram) else {
                    return unavailable(
                        FeatureStatus::Inapplicable,
                        "matrix column pair is outside the rank-2 diffusion chain".into(),
                    );
                };
                let base = (first * n + second) * self.spec.channels;
                for time_index in 0..4 {
                    let raw = profile.raw(time_index, class);
                    let stationary = profile.stationary[class];
                    let exact_zero = profile.exact_zero[time_index * profile.states.len() + class];
                    let underflow =
                        (raw == 0.0 && !exact_zero) || (raw > 0.0 && (raw as f32) == 0.0);
                    let offset = base + time_index * 7;
                    values[offset] = raw as f32;
                    values[offset + 1] = profile.log_class_size[class] as f32;
                    values[offset + 2] = stationary as f32;
                    values[offset + 3] = (raw.max(self.cache.config.floor).ln()
                        - stationary.max(self.cache.config.floor).ln())
                        as f32;
                    values[offset + 4] = f32::from(exact_zero);
                    values[offset + 5] = f32::from(underflow);
                    values[offset + 6] = f32::from(raw < self.cache.config.floor);
                }
            }
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, n, self.spec.channels],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

fn unavailable(status: FeatureStatus, reason: String) -> FeatureComputation {
    FeatureComputation::Unavailable { status, reason }
}

fn row_weights(state: &MatrixState) -> Vec<usize> {
    let words = state.words_per_row() as usize;
    state
        .packed_rows()
        .chunks_exact(words)
        .map(|row| row.iter().map(|word| word.count_ones() as usize).sum())
        .collect()
}

fn column_weights(state: &MatrixState) -> Vec<usize> {
    let n = state.dimension() as usize;
    (0..n)
        .map(|column| {
            (0..n)
                .map(|row| matrix_bit(state, row, column) as usize)
                .sum()
        })
        .collect()
}

fn matrix_bit(state: &MatrixState, row: usize, column: usize) -> bool {
    let words_per_row = state.words_per_row() as usize;
    let word = state.packed_rows()[row * words_per_row + column / 64];
    ((word >> (column % 64)) & 1) != 0
}

fn pair_histogram(state: &MatrixState, first: usize, second: usize) -> [usize; 4] {
    let n = state.dimension() as usize;
    let mut histogram = [0usize; 4];
    for row in 0..n {
        let pattern = (matrix_bit(state, row, first) as usize)
            | ((matrix_bit(state, row, second) as usize) << 1);
        histogram[pattern] += 1;
    }
    histogram
}

struct ProfileCache {
    config: DiffusionConfig,
    profiles: Mutex<HashMap<usize, Arc<ColumnProfile>>>,
    pair_profiles: Mutex<HashMap<usize, Arc<PairProfile>>>,
    mc_profiles: Mutex<HashMap<McKey, Arc<JointMoments>>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct McKey {
    dimension: u32,
    rows: Vec<u64>,
    seed: u64,
}

impl ProfileCache {
    fn get(&self, n: usize) -> Option<Arc<ColumnProfile>> {
        {
            let profiles = self
                .profiles
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(profile) = profiles.get(&n) {
                return Some(Arc::clone(profile));
            }
        }

        let profile = Arc::new(ColumnProfile::build(n, self.config)?);
        let mut profiles = self
            .profiles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = profiles.get(&n) {
            return Some(Arc::clone(existing));
        }
        if profiles.len() >= MAX_CACHED_WIDTHS {
            profiles.clear();
        }
        profiles.insert(n, Arc::clone(&profile));
        Some(profile)
    }

    fn get_pair(&self, n: usize) -> Option<Arc<PairProfile>> {
        {
            let profiles = self
                .pair_profiles
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(profile) = profiles.get(&n) {
                return Some(Arc::clone(profile));
            }
        }

        let profile = Arc::new(PairProfile::build(n, self.config)?);
        let mut profiles = self
            .pair_profiles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = profiles.get(&n) {
            return Some(Arc::clone(existing));
        }
        if profiles.len() >= MAX_CACHED_WIDTHS / 2 {
            profiles.clear();
        }
        profiles.insert(n, Arc::clone(&profile));
        Some(profile)
    }

    fn get_mc(&self, state: &MatrixState, seed: u64) -> Option<Arc<JointMoments>> {
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

struct JointMoments {
    samples: usize,
    total_gates: u64,
    mean_weight: Vec<f64>,
    covariance_weight: Vec<f64>,
    covariance_weight_intersection: Vec<f64>,
    covariance_weight_xor: Vec<f64>,
    mean_intersection: Vec<f64>,
    mean_xor: Vec<f64>,
    mean_character_pair: Vec<f64>,
    covariance_character_pair: Vec<f64>,
    standard_error_intersection: Vec<f64>,
    standard_error_xor: Vec<f64>,
    standard_error_character_pair: Vec<f64>,
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
