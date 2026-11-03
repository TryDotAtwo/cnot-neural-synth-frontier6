//! Validate and register the contextual rank feature families.
use super::{ContextualRankContext, ContextualRankProducer};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::rank_features::contexts::{RankIncrementProbe, RestrictedIntersection};
use crate::rank_features::providers::helpers::ensure_unconfigured;
use crate::schema::{FeatureId, FeatureSpec, feature_spec};
use crate::state::CnotGate;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

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

/// Registers C10 for explicit physical CNOT actions in the supplied alphabet.
pub fn register_selected_residual_ranks(
    registry: &mut FeatureRegistry,
    alphabet: &CnotAlphabet,
    gates: Vec<CnotGate>,
) -> Result<(), FeatureBuildError> {
    if gates.is_empty() {
        return Err(FeatureBuildError(
            "at least one selected physical CNOT is required".into(),
        ));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(gates.len())
        .map_err(|_| FeatureBuildError("selected-gate validation allocation failed".into()))?;
    for gate in &gates {
        if alphabet.cost(*gate).is_none() {
            return Err(FeatureBuildError(
                "selected CNOT must belong to the supplied physical alphabet".into(),
            ));
        }
        if !seen.insert(*gate) {
            return Err(FeatureBuildError(
                "selected CNOT gates must be unique".into(),
            ));
        }
    }

    let id = FeatureId::new(b'C', 10).unwrap();
    ensure_unconfigured(registry, &[id])?;
    let mut digest = Sha256::new();
    digest.update(b"cnot103/selected-residual-ranks/v1\0");
    digest.update(alphabet.dimension().to_le_bytes());
    digest.update((alphabet.id().len() as u64).to_le_bytes());
    digest.update(alphabet.id().as_bytes());
    digest.update((alphabet.gates().len() as u64).to_le_bytes());
    for gate in alphabet.gates() {
        digest.update(gate.control.to_le_bytes());
        digest.update(gate.target.to_le_bytes());
        digest.update(
            alphabet
                .cost(*gate)
                .expect("alphabet gates always have costs")
                .to_le_bytes(),
        );
    }
    digest.update((gates.len() as u64).to_le_bytes());
    for gate in &gates {
        digest.update(gate.control.to_le_bytes());
        digest.update(gate.target.to_le_bytes());
    }
    let version = format!("selected_residual_rank_v1:{}", finish_digest(digest));
    let spec = contextual_spec(
        id,
        version,
        "explicit_selected_physical_CNOT;rank_residual_not_distance",
        vec!["M".into(), "alphabet".into(), "selected_actions".into()],
        vec!["control".into(), "target".into()],
    );
    registry.register(Box::new(ContextualRankProducer {
        spec,
        context: ContextualRankContext::SelectedResidual {
            alphabet: alphabet.clone(),
            gates,
        },
    }))
}

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
