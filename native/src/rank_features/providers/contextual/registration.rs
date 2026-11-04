//! Validate and register the contextual rank feature families.

use super::{ContextualRankContext, ContextualRankProducer};
use crate::features::FeatureBuildError;
use crate::schema::{feature_spec, FeatureId, FeatureSpec};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

mod rank_increment;
mod restricted_intersections;
mod selected_residual;

pub use rank_increment::register_rank_increment_probes;
pub use restricted_intersections::register_restricted_intersections;
pub use selected_residual::register_selected_residual_ranks;

fn contextual_spec(
    id: FeatureId,
    producer_version: String,
    scope: &str,
    dependencies: Vec<String>,
    wire_roles: Vec<String>,
) -> FeatureSpec {
    let mut spec = feature_spec(id);
    spec.scope = scope.into();
    spec.producer_version = producer_version;
    spec.dependencies = dependencies;
    spec.exactness = "exact".into();
    spec.cost_tier = "T1".into();
    spec.wire_roles = wire_roles;
    spec
}

fn validate_unique_context<T: Eq + std::hash::Hash>(
    values: &[T],
    name: &str,
) -> Result<(), FeatureBuildError> {
    if values.is_empty() {
        return Err(FeatureBuildError(format!(
            "at least one {name} entry is required"
        )));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(values.len())
        .map_err(|_| FeatureBuildError(format!("{name} validation allocation failed")))?;
    if values.iter().any(|value| !seen.insert(value)) {
        return Err(FeatureBuildError(format!("{name} must be unique")));
    }
    Ok(())
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    digest.update((indices.len() as u64).to_le_bytes());
    for index in indices {
        digest.update(index.to_le_bytes());
    }
}

fn finish_digest(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
