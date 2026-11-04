//! F09/F10 proof-bound features backed by a verified general CNOT compiler.
//!
//! The archived schema has no legacy producers for these IDs. This provider
//! defines a versioned T2 contract: F09 exposes a certified lower/upper
//! interval, and F10 exposes the evidence class without treating it as model
//! confidence. A shortest-word claim is made only when the two bounds meet.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::general_compiler::GeneralCompilerLimits;
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

mod evidence;

const CACHE_CAPACITY: usize = 16;
const PRODUCER_VERSION: &str = "general_compiler_bounds_f09_f10_v1";

#[derive(Clone)]
struct BoundEvidence {
    f09: FeatureValue,
    f10: FeatureValue,
    provenance: Vec<String>,
}

#[derive(Clone)]
enum CachedBoundEvidence {
    Computed(BoundEvidence),
    Unavailable(FeatureStatus, String),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BoundCacheKey {
    state: StateIdentity,
}

struct BoundFeatureCache {
    limits: GeneralCompilerLimits,
    entries: Mutex<VecDeque<(BoundCacheKey, CachedBoundEvidence)>>,
}

impl BoundFeatureCache {
    fn get(&self, state: &MatrixState) -> CachedBoundEvidence {
        let key = BoundCacheKey {
            state: StateIdentity::from(state),
        };
        let Ok(mut entries) = self.entries.lock() else {
            return CachedBoundEvidence::Unavailable(
                FeatureStatus::ResourceLimit,
                "general_compiler_cache_poisoned".into(),
            );
        };
        if let Some(index) = entries.iter().position(|(cached, _)| cached == &key) {
            let entry = entries
                .remove(index)
                .expect("located general compiler cache entry");
            let result = entry.1.clone();
            entries.push_back(entry);
            return result;
        }

        let result = evidence::compute_bound_evidence(state, self.limits);
        if entries.len() == CACHE_CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, result.clone()));
        result
    }
}

struct BoundFeatureProducer {
    id: FeatureId,
    spec: FeatureSpec,
    cache: Arc<BoundFeatureCache>,
}

impl FeatureProducer for BoundFeatureProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != self.symmetry_contract() {
            return unavailable(
                FeatureStatus::Unsupported,
                "general_compiler_bounds_require_augmentation_only_symmetry",
            );
        }
        match self.cache.get(state) {
            CachedBoundEvidence::Computed(evidence) => {
                let value = if self.id == FeatureId::new(b'F', 9).unwrap() {
                    evidence.f09
                } else {
                    evidence.f10
                };
                FeatureComputation::ComputedWithProvenance {
                    value,
                    provenance: evidence.provenance,
                }
            }
            CachedBoundEvidence::Unavailable(status, reason) => {
                FeatureComputation::Unavailable { status, reason }
            }
        }
    }
}

/// Adds the context-free, width- and gate-bounded general compiler providers.
/// The bounded policy is explicit in producer identity and remains solver-gated.
pub fn register_general_compiler_bounds(
    registry: &mut FeatureRegistry,
) -> Result<(), FeatureBuildError> {
    let limits = GeneralCompilerLimits::default();
    let cache = Arc::new(BoundFeatureCache {
        limits,
        entries: Mutex::new(VecDeque::new()),
    });
    for (ordinal, channels, exactness) in [
        (
            9,
            4,
            "verified_upper_bound;certified_lower_bound;gap;bound_tightness",
        ),
        (
            10,
            5,
            "upper_lower_proof_flags;optimality_only_if_bounds_meet;search_completeness",
        ),
    ] {
        let id = FeatureId::new(b'F', ordinal).expect("F09 and F10 are canonical IDs");
        let mut spec = feature_spec(id);
        spec.role = FeatureRole::Global;
        spec.channels = channels;
        spec.scope = "full_matrix;all_to_all_unit_cnot;bounded_gauss_jordan".into();
        spec.producer_version = format!(
            "{PRODUCER_VERSION}:max_dimension={}:max_gates={}",
            limits.max_dimension, limits.max_gates
        );
        spec.dependencies = vec![
            "M".into(),
            "verified_general_cnot_word".into(),
            "certified_elementary_lower_bound".into(),
        ];
        spec.exactness = exactness.into();
        spec.cost_tier = "T2".into();
        spec.wire_roles.clear();
        spec.requires_solver_assisted = true;
        registry.register(Box::new(BoundFeatureProducer {
            id,
            spec,
            cache: Arc::clone(&cache),
        }))?;
    }
    Ok(())
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
