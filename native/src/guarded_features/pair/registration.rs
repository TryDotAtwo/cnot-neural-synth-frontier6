//! Registration metadata for the F02 and F06 verified pair views.

use super::super::{guarded_producer_version, CACHE_CAPACITY};
use super::GuardedPairProducer;
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use std::collections::VecDeque;
use std::sync::Mutex;

pub fn register_guarded_pair_f02_view(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F02 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 2).expect("F02 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 1;
    spec.scope = "solver_assisted;stage35_verified_pair_scope_only".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    // The Python reference inherits unused default roles for global features;
    // the native packet schema keeps incidence roles group-only.
    spec.wire_roles.clear();
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedPairProducer {
        spec,
        max_n,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

pub fn register_guarded_pair(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F06 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 6).expect("F06 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = 6;
    spec.scope = "solver_assisted;stage35_verified_pair_scope_only".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    spec.wire_roles = vec![
        "active_first".into(),
        "active_second".into(),
        "external_sources".into(),
    ];
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedPairProducer {
        spec,
        max_n,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}
