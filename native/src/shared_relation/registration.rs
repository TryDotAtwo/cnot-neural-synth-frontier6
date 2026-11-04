use super::context::SharedRelationContext;
use super::producer::SharedRelationProducer;
use super::{
    DEFAULT_MAX_COORDINATES, DEFAULT_MAX_PORTS, DEFAULT_MAX_VECTORS, EXACT_F32_INTEGER_LIMIT,
    STAGE56_SOURCE_SHA256,
};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Registers I06 for explicit, labelled physical contexts.
pub fn register_shared_relation_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<SharedRelationContext>,
    max_vectors: usize,
    max_coordinates: usize,
    max_ports: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty()
        || max_vectors == 0
        || max_coordinates == 0
        || max_ports == 0
        || max_vectors > EXACT_F32_INTEGER_LIMIT
        || max_ports > EXACT_F32_INTEGER_LIMIT
        || max_vectors.checked_add(2).is_none()
    {
        return Err(FeatureBuildError(
            "I06 requires contexts and positive finite budgets with exact float labels".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I06 allows one labelled context per physical width".into(),
        ));
    }
    let producer_version = producer_version(&contexts, max_vectors, max_coordinates, max_ports);
    let id = FeatureId::new(b'I', 6).expect("I06 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = max_vectors + 2;
    let spec = spec.with_provider_metadata(
            crate::schema::ProviderMetadata::new(
                "selected_physical_row_vectors;full_labelled_kernel_K_and_UX_bits;fixed_independent_U;no_rank_only_replacement",
                producer_version,
                "exact",
                "T2",
            )
            .dependencies(["M", "labelled_rows", "ambient_coordinates", "covector_frame"])
            .wire_roles(["vector_row", "covector_coordinate"]),
        );
    registry.register(Box::new(SharedRelationProducer {
        spec,
        contexts,
        max_vectors,
        max_coordinates,
        max_ports,
    }))
}

pub fn register_default_shared_relation_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<SharedRelationContext>,
) -> Result<(), FeatureBuildError> {
    register_shared_relation_contexts(
        registry,
        contexts,
        DEFAULT_MAX_VECTORS,
        DEFAULT_MAX_COORDINATES,
        DEFAULT_MAX_PORTS,
    )
}

fn producer_version(
    contexts: &[SharedRelationContext],
    max_vectors: usize,
    max_coordinates: usize,
    max_ports: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"stage56_full_shared_relation_v1");
    hash_field(&mut digest, STAGE56_SOURCE_SHA256.as_bytes());
    hash_field(&mut digest, &max_vectors.to_le_bytes());
    hash_field(&mut digest, &max_coordinates.to_le_bytes());
    hash_field(&mut digest, &max_ports.to_le_bytes());
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_indices(&mut digest, &context.rows);
        hash_indices(&mut digest, &context.coordinates);
        hash_field(&mut digest, &(context.covectors.len() as u64).to_le_bytes());
        for covector in &context.covectors {
            hash_field(&mut digest, &(covector.len() as u64).to_le_bytes());
            for word in covector {
                hash_field(&mut digest, &word.to_le_bytes());
            }
        }
    }
    format!(
        "stage56_full_shared_relation_v1:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    hash_field(digest, &(indices.len() as u64).to_le_bytes());
    for index in indices {
        hash_field(digest, &index.to_le_bytes());
    }
}
