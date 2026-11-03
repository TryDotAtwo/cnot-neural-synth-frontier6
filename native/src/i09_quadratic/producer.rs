use super::evaluation::compute_i09_response;
use super::types::{context_digest, I09Context, I09ErrorKind, I09Limits};
use super::I09_STAGE60_SOURCE_SHA256;
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

#[derive(Clone)]
pub(super) struct I09Producer {
    pub(super) spec: FeatureSpec,
    pub(super) contexts: Vec<I09Context>,
    pub(super) limits: I09Limits,
}

impl FeatureProducer for I09Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(FeatureStatus::Unsupported, "i09_requires_augmentation_only");
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension() as usize)
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i09_context_for_matrix_width",
            );
        };
        let expected_alphabet = format!("all_to_all_unit_cnot:{}", context.n);
        if state.alphabet_id() != expected_alphabet {
            return unavailable(
                FeatureStatus::Unsupported,
                "i09_requires_matching_all_to_all_unit_cnot_alphabet",
            );
        }
        match compute_i09_response(state, context, self.limits) {
            Ok(response) => match response.to_feature_value(self.limits, context.n) {
                Ok(value) => FeatureComputation::ComputedWithProvenance {
                    value,
                    provenance: vec![format!(
                        "stage60_source_sha256={};context_sha256={}",
                        I09_STAGE60_SOURCE_SHA256,
                        context_digest(context)
                    )],
                },
                Err(error) => unavailable(
                    FeatureStatus::ResourceLimit,
                    &format!("i09_output_conversion_failed:{}", error.reason),
                ),
            },
            Err(error) => unavailable(
                match error.kind {
                    I09ErrorKind::Unsupported => FeatureStatus::Unsupported,
                    I09ErrorKind::ResourceLimit | I09ErrorKind::Internal => {
                        FeatureStatus::ResourceLimit
                    }
                },
                &error.reason,
            ),
        }
    }
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
