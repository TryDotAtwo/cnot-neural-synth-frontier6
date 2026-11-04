//! Exact finite-state diffusion feature producer for G01-G04 and G06.

mod column;
mod pair;
mod weights;

use super::ProfileCache;
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::schema::FeatureSpec;
use crate::state::MatrixState;
use std::sync::Arc;

pub(super) struct DiffusionProducer {
    spec: FeatureSpec,
    cache: Arc<ProfileCache>,
}

impl FeatureProducer for DiffusionProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        if self.spec.id.ordinal() == 6 {
            pair::compute(self, state, n)
        } else {
            column::compute(self, state, n)
        }
    }
}

impl DiffusionProducer {
    pub(super) fn new(spec: FeatureSpec, cache: Arc<ProfileCache>) -> Self {
        Self { spec, cache }
    }
}
