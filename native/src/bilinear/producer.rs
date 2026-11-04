mod reduction;
mod registration;

pub(super) use reduction::reduce_stage55;
pub use registration::{register_bilinear_contexts, register_default_bilinear_contexts};

use super::context::BilinearContext;
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureValue};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

struct BilinearProducer {
    spec: FeatureSpec,
    contexts: Vec<BilinearContext>,
    max_dimension: usize,
    max_matrix_bits: usize,
}

impl FeatureProducer for BilinearProducer {
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
                "i05_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i05_context_for_matrix_width",
            );
        };

        let (m, r, p, t, q) = context.dimensions();
        let largest = [m, r, p, t, q].into_iter().max().unwrap_or_default();
        if largest > self.max_dimension
            || context
                .matrix_bit_budget()
                .is_none_or(|bits| bits > self.max_matrix_bits)
        {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i05_dimension_or_matrix_bit_budget_exceeded",
            );
        }
        if policy.max_bytes.is_some_and(|limit| {
            context
                .working_bytes_upper_bound()
                .is_none_or(|estimate| estimate > limit)
        }) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i05_working_memory_budget_exceeded_before_matrix_allocation",
            );
        }

        match context.source_values(state) {
            Ok(values) if values.iter().all(|value| *value <= (1 << 24)) => {
                FeatureComputation::Computed(FeatureValue {
                    shape: vec![2],
                    values: values.into_iter().map(|value| value as f32).collect(),
                    token_valid: Vec::new(),
                    incidence: Vec::new(),
                })
            }
            Ok(_) => unavailable(
                FeatureStatus::ResourceLimit,
                "i05_integer_value_exceeds_exact_f32_range",
            ),
            Err(reason) => unavailable(FeatureStatus::ResourceLimit, &reason),
        }
    }
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
