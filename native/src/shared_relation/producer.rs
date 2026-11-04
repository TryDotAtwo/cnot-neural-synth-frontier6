use super::context::SharedRelationContext;
use super::DEFAULT_MAX_BYTES;

mod compute;
mod memory;
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

pub(super) struct SharedRelationProducer {
    pub(super) spec: FeatureSpec,
    pub(super) contexts: Vec<SharedRelationContext>,
    pub(super) max_vectors: usize,
    pub(super) max_coordinates: usize,
    pub(super) max_ports: usize,
}

impl FeatureProducer for SharedRelationProducer {
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
                "i06_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i06_context_for_matrix_width",
            );
        };
        let (n, vectors, coordinates, ports) = (
            usize::try_from(context.n).unwrap_or(usize::MAX),
            context.rows.len(),
            context.coordinates.len(),
            context.covectors.len(),
        );
        if vectors > self.max_vectors
            || coordinates > self.max_coordinates
            || ports > self.max_ports
        {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i06_context_exceeds_declared_vector_coordinate_or_port_budget",
            );
        }
        let Some(estimate) =
            memory::working_bytes_upper_bound(n, vectors, coordinates, ports, self.max_vectors)
        else {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i06_working_memory_size_overflow",
            );
        };
        if estimate > policy.max_bytes.unwrap_or(DEFAULT_MAX_BYTES) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i06_working_memory_budget_exceeded_before_allocation",
            );
        }

        match compute::compute_tokens(state, context, self.max_vectors, estimate) {
            Ok(value) => FeatureComputation::Computed(value),
            Err(reason) => unavailable(FeatureStatus::ResourceLimit, reason),
        }
    }
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
