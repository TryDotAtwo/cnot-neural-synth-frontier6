//! Opt-in source-relative CNOT diffusion features G01-G04 and G06-G10.
//!
//! The source computes finite continuous-time Markov semigroups over exact
//! column-weight and rank-2 pair-histogram classes. Uniformization uses the same transition counts and
//! retains the source's floating-semigroup status; it does not claim a formal
//! numerical error certificate.

use crate::features::{FeatureBuildError, FeatureComputation, FeatureRegistry};
use crate::packet::FeatureStatus;
use crate::schema::{FeatureId, feature_spec};
use sha2::{Digest, Sha256};
use std::sync::Arc;

mod cache;
mod deterministic;
mod monte_carlo;
mod profiles;
use cache::ProfileCache;
use deterministic::DiffusionProducer;
use monte_carlo::MonteCarloProducer;

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
    let cache = Arc::new(ProfileCache::new(config));
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
        registry.register(Box::new(DiffusionProducer::new(spec, Arc::clone(&cache))))?;
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
        registry.register(Box::new(MonteCarloProducer::new(spec, Arc::clone(&cache))))?;
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

fn unavailable(status: FeatureStatus, reason: String) -> FeatureComputation {
    FeatureComputation::Unavailable { status, reason }
}
