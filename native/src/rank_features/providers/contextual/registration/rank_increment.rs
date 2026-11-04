use super::{
    contextual_spec, finish_digest, hash_indices, validate_unique_context, ContextualRankContext,
    ContextualRankProducer,
};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::rank_features::contexts::RankIncrementProbe;
use crate::rank_features::providers::helpers::ensure_unconfigured;
use crate::schema::FeatureId;
use sha2::{Digest, Sha256};

/// Registers C09 for explicit restricted spans and candidate parity vectors.
pub fn register_rank_increment_probes(
    registry: &mut FeatureRegistry,
    probes: Vec<RankIncrementProbe>,
) -> Result<(), FeatureBuildError> {
    validate_unique_context(&probes, "rank-increment probes")?;
    let id = FeatureId::new(b'C', 9).unwrap();
    ensure_unconfigured(registry, &[id])?;
    let mut digest = Sha256::new();
    digest.update(b"cnot103/restricted-rank-increments/v1\0");
    digest.update((probes.len() as u64).to_le_bytes());
    for probe in &probes {
        digest.update(probe.n.to_le_bytes());
        hash_indices(&mut digest, &probe.span_rows);
        hash_indices(&mut digest, &probe.coordinates);
        hash_indices(&mut digest, &probe.candidate_support);
    }
    let version = format!("restricted_rank_increment_v1:{}", finish_digest(digest));
    let spec = contextual_spec(
        id,
        version,
        "explicit_restricted_span_and_candidate;augmentation_only",
        vec![
            "M".into(),
            "restricted_index_context".into(),
            "candidate_parity".into(),
        ],
        vec![
            "span_rows".into(),
            "coordinate_slice".into(),
            "candidate_support".into(),
        ],
    );
    registry.register(Box::new(ContextualRankProducer {
        spec,
        context: ContextualRankContext::RankIncrement(probes),
    }))
}
