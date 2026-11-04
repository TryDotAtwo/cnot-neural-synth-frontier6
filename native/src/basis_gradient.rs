//! Exact zero-template partial-shear cost and gradient descriptors.
//!
//! These are augmentation-only because the source construction fixes the first
//! half of the physical wires as its source block. They describe the selected
//! partial elimination; they are not global lower bounds or optimality claims.

use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

mod compute;
mod registration;
pub use registration::register_basis_gradient;

#[derive(Clone, Copy)]
enum BasisGradientKind {
    Cost,
    PairGradient,
    GradientStatistics,
}

struct BasisGradientProducer {
    spec: FeatureSpec,
    kind: BasisGradientKind,
}

impl FeatureProducer for BasisGradientProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != self.symmetry_contract() {
            return unavailable("basis_gradient_requires_augmentation_only_symmetry");
        }
        compute::compute(self.kind, state)
    }
}

fn unavailable(reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status: FeatureStatus::Unsupported,
        reason: reason.into(),
    }
}

fn inapplicable(reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status: FeatureStatus::Inapplicable,
        reason: reason.into(),
    }
}

fn resource_limit(reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status: FeatureStatus::ResourceLimit,
        reason: reason.into(),
    }
}
