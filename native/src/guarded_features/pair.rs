//! Verified pair-scope feature producers backed by the stage-35 recognizer.

mod cache;
mod compute;
mod registration;

pub use registration::{register_guarded_pair, register_guarded_pair_f02_view};

use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::schema::FeatureSpec;
use crate::stage35::Stage35Result;
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::Mutex;

struct CachedPair {
    state: MatrixState,
    result: Stage35Result,
}

struct GuardedPairProducer {
    spec: FeatureSpec,
    max_n: u32,
    cache: Mutex<VecDeque<CachedPair>>,
}

impl FeatureProducer for GuardedPairProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        if self.spec.id.family() == 'F' && self.spec.id.ordinal() == 2 {
            "exact_relabel"
        } else {
            "augmentation_only"
        }
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        compute::compute(self, state, policy)
    }
}
