//! Registry validation and public registration entry points.

use super::super::compute::factor_entry_count;
use super::super::{
    OptionalGatePacket, DEFAULT_MAX_FACTOR_ENTRIES, DEFAULT_MAX_MONOMIALS, DEFAULT_MAX_N,
    DEFAULT_MAX_SLOTS, MAX_EXACT_F32_INTEGER,
};
use super::identity::producer_version;
use super::provider::PositivePacketProducer;
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use std::collections::HashSet;

/// Registers I08 for explicit, labelled optional-CNOT contexts.
pub fn register_positive_packet_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<OptionalGatePacket>,
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty()
        || max_n == 0
        || max_slots == 0
        || max_monomials == 0
        || max_factor_entries == 0
        || max_slots > MAX_EXACT_F32_INTEGER / 2
    {
        return Err(FeatureBuildError(
            "I08 requires contexts and positive budgets with exact float labels".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts
        .iter()
        .any(|context| !widths.insert(context.n) || factor_entry_count(&context.factors).is_none())
    {
        return Err(FeatureBuildError(
            "I08 allows one valid labelled context per physical width".into(),
        ));
    }
    let producer_version = producer_version(
        &contexts,
        max_n,
        max_slots,
        max_monomials,
        max_factor_entries,
    );
    let id = FeatureId::new(b'I', 8).expect("I08 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = 4;
    let spec = spec.with_provider_metadata(
            crate::schema::ProviderMetadata::new(
                "fixed_order_affine_optional_CNOT_packet;explicit_local_factors;cut,kappa,live,log2_state_upper_bound;not_distance",
                producer_version,
                "exact",
                "T2",
            )
            .dependencies(["M", "optional_physical_gate_slots", "local_factor_truth_tables"])
            .wire_roles(["control", "target"]),
        );
    registry.register(Box::new(PositivePacketProducer {
        spec,
        contexts,
        max_n,
        max_slots,
        max_monomials,
        max_factor_entries,
    }))
}

pub fn register_default_positive_packet_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<OptionalGatePacket>,
) -> Result<(), FeatureBuildError> {
    register_positive_packet_contexts(
        registry,
        contexts,
        DEFAULT_MAX_N,
        DEFAULT_MAX_SLOTS,
        DEFAULT_MAX_MONOMIALS,
        DEFAULT_MAX_FACTOR_ENTRIES,
    )
}
