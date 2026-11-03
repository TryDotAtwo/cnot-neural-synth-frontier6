//! Validate physical contexts and register the shared D09-D14 providers.
use super::cache::{PaidBasisFeatureCache, PaidBasisFeatureProducer};
use super::helpers::{config_digest, d_channels};
use super::{D_IDS, MAX_CONTEXTS, PaidBasisBlockContext, WIRE_ROLES};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::paid_basis::PaidShearConfig;
use crate::schema::{FeatureId, FeatureRole, feature_spec};
use std::collections::HashSet;
use std::sync::Arc;

/// Registers the six solver-assisted paid-basis features for explicit physical
/// blocks. Group features carry source/target incidence; D14 is a pair tensor
/// and exposes its endpoint digest only through packet provenance.
pub fn register_paid_basis_features(
    registry: &mut FeatureRegistry,
    contexts: Vec<PaidBasisBlockContext>,
    config: PaidShearConfig,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || contexts.len() > MAX_CONTEXTS {
        return Err(FeatureBuildError(format!(
            "paid-basis registration requires 1..={MAX_CONTEXTS} explicit contexts"
        )));
    }
    if !(2..=4).contains(&config.local_width) || config.sweeps > 16 {
        return Err(FeatureBuildError(
            "paid-basis local width must be 2..=4 and sweep count at most 16".into(),
        ));
    }
    let mut seen = HashSet::new();
    if contexts.iter().any(|context| !seen.insert(context.clone())) {
        return Err(FeatureBuildError(
            "paid-basis physical contexts must be unique".into(),
        ));
    }

    let specs = registry.specs();
    for ordinal in D_IDS {
        let id = FeatureId::new(b'D', ordinal).expect("D09-D14 are canonical feature IDs");
        if specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let version = format!(
        "verified_paid_basis_d09_d14_v1:{}",
        config_digest(&contexts, config)
    );
    let shared = Arc::new(PaidBasisFeatureCache::new(contexts, config));
    for ordinal in D_IDS {
        let id = FeatureId::new(b'D', ordinal).expect("D09-D14 are canonical feature IDs");
        let mut spec = feature_spec(id);
        spec.role = if ordinal == 14 {
            FeatureRole::Pair
        } else {
            FeatureRole::Group
        };
        spec.channels = d_channels(ordinal);
        spec.scope =
            "explicit_physical_blocks;verified_paid_basis_candidate;augmentation_only".into();
        spec.producer_version = version.clone();
        spec.dependencies = vec![
            "M".into(),
            "explicit_physical_blocks".into(),
            "exact_local_GL4_atlas".into(),
            "full_physical_CNOT_word_verifier".into(),
        ];
        spec.exactness =
            "exact_local_candidate;full_endpoint_replayed;not_global_optimality".into();
        spec.cost_tier = "T2".into();
        spec.wire_roles = if ordinal == 14 {
            Vec::new()
        } else {
            WIRE_ROLES.iter().map(|role| (*role).into()).collect()
        };
        spec.requires_solver_assisted = true;
        registry.register(Box::new(PaidBasisFeatureProducer {
            id,
            spec,
            shared: Arc::clone(&shared),
        }))?;
    }
    Ok(())
}
