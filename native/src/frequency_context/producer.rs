use super::fingerprint::producer_version;
use super::support::{gf2_rank, has_stage52_interface, unavailable};
use super::types::FrequencyContext;
use super::{DEFAULT_MAX_FREQUENCY_BITS, DEFAULT_MAX_WIDTH, MAX_EXACT_F32_INTEGER};
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use std::collections::HashSet;
use std::sync::Mutex;

struct FrequencyContextProducer {
    spec: FeatureSpec,
    contexts: Vec<FrequencyContext>,
    max_width: u32,
    max_frequency_bits: usize,
    rank_cache: Mutex<std::collections::HashMap<u32, usize>>,
}

impl FeatureProducer for FrequencyContextProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "i07_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "i07_no_explicit_frequency_context_for_width",
            );
        };
        let expected_alphabet = format!("all_to_all_unit_cnot:{}", context.n);
        if state.alphabet_id() != expected_alphabet || state.dimension() != context.n {
            return unavailable(FeatureStatus::Unsupported, "i07_state_alphabet_mismatch");
        }

        let Some(frequency_bits) = context.sources.len().checked_mul(context.targets.len()) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_frequency_bit_count_overflow",
            );
        };
        if context.n > self.max_width || frequency_bits > self.max_frequency_bits {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_explicit_context_exceeds_configured_budget",
            );
        }
        match has_stage52_interface(state, context) {
            Ok(true) => {}
            Ok(false) => {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "i07_state_outside_explicit_stage52_interface",
                );
            }
            Err(()) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "i07_interface_rank_allocation_failed",
                );
            }
        }

        let Some(kappa) = self.rank_for(context) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_central_rank_allocation_failed",
            );
        };
        let Some(representation_exponent) = context.internal.len().checked_mul(kappa) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_representation_exponent_overflow",
            );
        };
        if kappa > MAX_EXACT_F32_INTEGER || representation_exponent > MAX_EXACT_F32_INTEGER {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i07_feature_value_exceeds_exact_float32_integer_range",
            );
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![2],
            values: vec![kappa as f32, representation_exponent as f32],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl FrequencyContextProducer {
    fn rank_for(&self, context: &FrequencyContext) -> Option<usize> {
        match self.rank_cache.lock() {
            Ok(mut cache) => {
                if let Some(rank) = cache.get(&context.n) {
                    return Some(*rank);
                }
                let rank = gf2_rank(&context.central_rows, context.sources.len())?;
                cache.insert(context.n, rank);
                Some(rank)
            }
            Err(_) => gf2_rank(&context.central_rows, context.sources.len()),
        }
    }
}

/// Registers I07 for explicit labelled interface contexts, with source budgets.
///
/// The default limits match the Python reference's width and `r*m` frequency
/// budget. Contexts for distinct widths are required; their wire order is kept
/// exactly as supplied and is not relabel-invariant.
pub fn register_frequency_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<FrequencyContext>,
    max_width: u32,
    max_frequency_bits: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || max_width == 0 || max_frequency_bits == 0 {
        return Err(FeatureBuildError(
            "I07 requires explicit contexts and positive width/frequency budgets".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I07 allows at most one labelled Fourier context per width".into(),
        ));
    }

    let producer_version = producer_version(&contexts, max_width, max_frequency_bits);
    let id = FeatureId::new(b'I', 7).expect("I07 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 2;
    let spec = spec.with_provider_metadata(
        concat!(
            "explicit_full_stage52_interface;fixed_labelled_central_Fourier_sector;",
            "kappa_and_log2_representation_dimension;not_distance"
        ),
        producer_version,
        [
            "M",
            "physical_interface_partition",
            "central_frequency_matrix",
        ],
        "exact",
        "T2",
        std::iter::empty::<String>(),
        false,
    );
    registry.register(Box::new(FrequencyContextProducer {
        spec,
        contexts,
        max_width,
        max_frequency_bits,
        rank_cache: Mutex::new(std::collections::HashMap::new()),
    }))
}

pub fn register_default_frequency_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<FrequencyContext>,
) -> Result<(), FeatureBuildError> {
    register_frequency_contexts(
        registry,
        contexts,
        DEFAULT_MAX_WIDTH,
        DEFAULT_MAX_FREQUENCY_BITS,
    )
}
