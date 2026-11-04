//! H06 exact pair differences across physical matrix views.
//!
//! The provider follows research_v3.features.symmetry_alignment: for each
//! ordered physical wire pair it concatenates the differences in A15, A16,
//! and D03 descriptors for M^T, M^-1, and M^-T relative to M.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;

mod compute;
use std::collections::VecDeque;
use std::sync::Mutex;

const H06_CHANNELS: usize = 15;
const VIEW_COUNT: usize = 4;
const DESCRIPTOR_CHANNELS: usize = 5;
const CACHE_CAPACITY: usize = 32;

#[derive(Clone)]
struct CacheEntry {
    state: StateIdentity,
    max_bytes: Option<usize>,
    value: FeatureComputation,
}

struct H06Producer {
    spec: FeatureSpec,
    cache: Mutex<VecDeque<CacheEntry>>,
}

impl FeatureProducer for H06Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        let identity = StateIdentity::from(state);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache
                .iter()
                .position(|entry| entry.state == identity && entry.max_bytes == policy.max_bytes)
            {
                if let Some(entry) = cache.remove(index) {
                    let result = entry.value.clone();
                    cache.push_front(entry);
                    return result;
                }
            }
        }

        let value = compute::compute_h06(state, policy.max_bytes);
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CacheEntry {
                state: identity,
                max_bytes: policy.max_bytes,
                value: value.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        value
    }
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

/// Register H06's exact 15-channel physical pair descriptor.
pub fn register_symmetry_alignment_h06(
    registry: &mut FeatureRegistry,
) -> Result<(), FeatureBuildError> {
    let id = FeatureId::new(b'H', 6).expect("H06 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Pair;
    spec.channels = H06_CHANNELS;
    let spec = spec.with_provider_metadata(
        "physical_pair_differences_A15_A16_width2_D03_across_T_inverse_inverseT",
        "physical_symmetry_alignment_v1",
        ["M", "exact_GF2_inverse", "physical_wire_matching"],
        "exact",
        "T1",
        std::iter::empty::<&str>(),
        false,
    );
    registry.register(Box::new(H06Producer {
        spec,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}
