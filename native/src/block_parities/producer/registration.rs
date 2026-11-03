use super::compute::{compute_group_feature, compute_node_frequencies, unavailable};
use super::patterns::{parity_data, PreparedBlock};
use crate::block_parities::BlockContext;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

struct BlockParityProducer {
    spec: FeatureSpec,
    contexts: Vec<BlockContext>,
    max_patterns: usize,
    max_pairs: usize,
}

impl FeatureProducer for BlockParityProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "physical_block_parities_require_augmentation_only_symmetry",
            );
        }
        let relevant = self
            .contexts
            .iter()
            .filter(|context| context.n == state.dimension())
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_physical_block_context_for_matrix_width",
            );
        }
        if self.spec.id.ordinal() == 5 && relevant.len() != 1 {
            return unavailable(
                FeatureStatus::Unsupported,
                "node_block_frequency_requires_exactly_one_context",
            );
        }

        let mut prepared = Vec::new();
        if prepared.try_reserve_exact(relevant.len()).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "block context allocation failed",
            );
        }
        for context in relevant {
            let data = match parity_data(state, context, self.max_patterns) {
                Ok(data) => data,
                Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
            };
            let mut patterns = Vec::new();
            if patterns.try_reserve_exact(data.counts.len()).is_err() {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "pattern index allocation failed",
                );
            }
            patterns.extend(data.counts.keys().cloned());
            patterns.sort_unstable();
            prepared.push(PreparedBlock {
                context,
                data,
                patterns,
            });
        }

        if self.spec.id.ordinal() == 5 {
            return compute_node_frequencies(state, &prepared[0]);
        }
        compute_group_feature(
            state,
            self.spec.id.ordinal(),
            self.spec.channels,
            &self.spec.wire_roles,
            &prepared,
            self.max_pairs,
        )
    }
}

/// Registers D01/D02/D04-D08 for explicit physical source/target wire blocks.
pub fn register_block_parities(
    registry: &mut FeatureRegistry,
    contexts: Vec<BlockContext>,
    max_patterns: usize,
    max_pairs: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || max_patterns == 0 {
        return Err(FeatureBuildError(
            "explicit block contexts and a positive pattern budget are required".into(),
        ));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(contexts.len())
        .map_err(|_| FeatureBuildError("block-context validation allocation failed".into()))?;
    if contexts.iter().any(|context| !seen.insert(context)) {
        return Err(FeatureBuildError("block contexts must be unique".into()));
    }

    let ids = [1, 2, 4, 5, 6, 7, 8].map(|ordinal| {
        FeatureId::new(b'D', ordinal).expect("D01-D08 IDs are in the canonical feature manifest")
    });
    let specs = registry.specs();
    for id in &ids {
        if specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let mut digest = Sha256::new();
    digest.update(b"cnot103/physical-block-parities/v1\0");
    digest.update((contexts.len() as u64).to_le_bytes());
    for context in &contexts {
        digest.update(context.n.to_le_bytes());
        hash_indices(&mut digest, &context.sources);
        hash_indices(&mut digest, &context.targets);
    }
    digest.update((max_patterns as u64).to_le_bytes());
    digest.update((max_pairs as u64).to_le_bytes());
    let version = format!("physical_block_parities_v1:{}", finish_digest(digest));

    for id in ids {
        let ordinal = id.ordinal();
        let mut spec = feature_spec(id);
        spec.scope = "explicit_physical_block;sparse_exact_histogram;augmentation_only".into();
        spec.producer_version = version.clone();
        spec.dependencies = vec!["M".into(), "physical_source_target_context".into()];
        spec.exactness = "exact".into();
        spec.cost_tier = "T1".into();
        spec.wire_roles = if ordinal == 5 {
            Vec::new()
        } else if ordinal == 6 {
            vec![
                "source_coordinates".into(),
                "left_parity".into(),
                "right_parity".into(),
            ]
        } else {
            vec![
                "source_coordinates".into(),
                "parity_support".into(),
                "target_members".into(),
            ]
        };
        debug_assert!(matches!(spec.role, FeatureRole::Node | FeatureRole::Group));
        registry.register(Box::new(BlockParityProducer {
            spec,
            contexts: contexts.clone(),
            max_patterns,
            max_pairs,
        }))?;
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
