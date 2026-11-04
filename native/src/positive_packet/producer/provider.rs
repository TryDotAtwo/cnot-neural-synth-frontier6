//! Feature evaluation for explicit positive optional-CNOT packets.

use super::super::compute::{
    compute_values, factor_entry_count, working_bytes_upper_bound, ComputeFailure,
};
use super::super::{OptionalGatePacket, DEFAULT_MAX_BYTES};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

pub(super) struct PositivePacketProducer {
    pub(super) spec: FeatureSpec,
    pub(super) contexts: Vec<OptionalGatePacket>,
    pub(super) max_n: usize,
    pub(super) max_slots: usize,
    pub(super) max_monomials: usize,
    pub(super) max_factor_entries: usize,
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

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
