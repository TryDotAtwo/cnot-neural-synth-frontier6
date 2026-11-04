use super::{
    contextual_spec, finish_digest, hash_indices, validate_unique_context, ContextualRankContext,
    ContextualRankProducer,
};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::rank_features::contexts::RestrictedIntersection;
use crate::rank_features::providers::helpers::ensure_unconfigured;
use crate::schema::FeatureId;
use sha2::{Digest, Sha256};

/// Registers C06/C07 for explicit coordinate-restricted row/column subspaces.
pub fn register_restricted_intersections(
    registry: &mut FeatureRegistry,
    probes: Vec<RestrictedIntersection>,
) -> Result<(), FeatureBuildError> {
    validate_unique_context(&probes, "restricted-intersection probes")?;
    let ids = [
        FeatureId::new(b'C', 6).unwrap(),
        FeatureId::new(b'C', 7).unwrap(),
    ];
    ensure_unconfigured(registry, &ids)?;
    let mut digest = Sha256::new();
    digest.update(b"cnot103/restricted-intersections/v1\0");
    digest.update((probes.len() as u64).to_le_bytes());
    for probe in &probes {
        digest.update(probe.n.to_le_bytes());
        hash_indices(&mut digest, &probe.left);
        hash_indices(&mut digest, &probe.right);
        hash_indices(&mut digest, &probe.coordinates);
    }
    let version = format!("restricted_intersections_v1:{}", finish_digest(digest));
    for (id, transpose) in ids.into_iter().zip([false, true]) {
        let spec = contextual_spec(
            id,
            version.clone(),
            "explicit_restricted_subspaces;augmentation_only",
            vec!["M".into(), "restricted_index_context".into()],
            vec![
                "left_span".into(),
                "right_span".into(),
                "coordinate_slice".into(),
            ],
        );
        registry.register(Box::new(ContextualRankProducer {
            spec,
            context: ContextualRankContext::RestrictedIntersection {
                probes: probes.clone(),
                transpose,
            },
        }))?;
    }
    Ok(())
}
