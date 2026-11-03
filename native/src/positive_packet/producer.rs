use super::compute::{
    ComputeFailure, compute_values, factor_entry_count, working_bytes_upper_bound,
};
use super::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_FACTOR_ENTRIES, DEFAULT_MAX_MONOMIALS, DEFAULT_MAX_N,
    DEFAULT_MAX_SLOTS, MAX_EXACT_F32_INTEGER, OptionalGatePacket, STAGE59_SOURCE_SHA256,
};
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
};
use crate::packet::FeatureStatus;
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, feature_spec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

struct PositivePacketProducer {
    spec: FeatureSpec,
    contexts: Vec<OptionalGatePacket>,
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
}

impl FeatureProducer for PositivePacketProducer {
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
                "i08_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i08_context_for_matrix_width",
            );
        };
        let n = usize::try_from(context.n).unwrap_or(usize::MAX);
        let slots = context.gates.len();
        if n > self.max_n
            || slots > self.max_slots
            || factor_entry_count(&context.factors)
                .is_none_or(|count| count > self.max_factor_entries)
        {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i08_context_exceeds_declared_width_slot_or_factor_budget",
            );
        }
        let expected_alphabet = format!("all_to_all_unit_cnot:{n}");
        if state.alphabet_id() != expected_alphabet {
            return unavailable(
                FeatureStatus::Unsupported,
                "i08_requires_matching_all_to_all_unit_cnot_alphabet",
            );
        }
        let Some(estimate) = working_bytes_upper_bound(n, slots, self.max_monomials) else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i08_working_memory_size_overflow",
            );
        };
        if estimate > policy.max_bytes.unwrap_or(DEFAULT_MAX_BYTES) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i08_working_memory_budget_exceeded_before_allocation",
            );
        }

        match compute_values(state, context, self.max_monomials) {
            Ok(value) => FeatureComputation::Computed(value),
            Err(ComputeFailure::ResourceLimit) => unavailable(
                FeatureStatus::ResourceLimit,
                "i08_endpoint_polynomial_or_allocation_exceeded_declared_budget",
            ),
            Err(ComputeFailure::NonAffine) => {
                unavailable(FeatureStatus::Unsupported, "i08_endpoint_map_is_non_affine")
            }
            Err(ComputeFailure::Internal) => unavailable(
                FeatureStatus::Unsupported,
                "i08_internal_rank_or_shape_invariant_failed",
            ),
        }
    }
}

fn producer_version(
    contexts: &[OptionalGatePacket],
    max_n: usize,
    max_slots: usize,
    max_monomials: usize,
    max_factor_entries: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(
        &mut digest,
        b"stage59_optional_packet_v2_source_checked_cache",
    );
    hash_field(&mut digest, STAGE59_SOURCE_SHA256.as_bytes());
    for budget in [max_n, max_slots, max_monomials, max_factor_entries] {
        hash_field(&mut digest, &(budget as u64).to_le_bytes());
    }
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_field(&mut digest, &(context.gates.len() as u64).to_le_bytes());
        for (control, target) in &context.gates {
            hash_field(&mut digest, &control.to_le_bytes());
            hash_field(&mut digest, &target.to_le_bytes());
        }
        hash_field(&mut digest, &(context.factors.len() as u64).to_le_bytes());
        for factor in &context.factors {
            hash_field(&mut digest, &(factor.scope.len() as u64).to_le_bytes());
            for selector in &factor.scope {
                hash_field(&mut digest, &selector.to_le_bytes());
            }
            hash_field(&mut digest, &(factor.allowed.len() as u64).to_le_bytes());
            for pattern in &factor.allowed {
                hash_field(&mut digest, &pattern.to_bytes_le());
            }
        }
    }
    format!(
        "stage59_optional_packet_v2_source_checked_cache:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

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
        "fixed_order_affine_optional_CNOT_packet;explicit_local_factors;cut,kappa,live,log2_state_upper_bound;not_distance",
        producer_version,
        ["M", "optional_physical_gate_slots", "local_factor_truth_tables"],
        "exact",
        "T2",
        ["control", "target"],
        false,
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
