use super::super::fingerprint::producer_version;
use super::super::types::FrequencyContext;
use super::super::{DEFAULT_MAX_FREQUENCY_BITS, DEFAULT_MAX_WIDTH};
use super::FrequencyContextProducer;
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use std::collections::HashSet;
use std::sync::Mutex;

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
        crate::schema::ProviderMetadata::new(
            concat!(
                "explicit_full_stage52_interface;fixed_labelled_central_Fourier_sector;",
                "kappa_and_log2_representation_dimension;not_distance"
            ),
            producer_version,
            "exact",
            "T2",
        )
        .dependencies([
            "M",
            "physical_interface_partition",
            "central_frequency_matrix",
        ])
        .wire_roles(std::iter::empty::<String>()),
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
