use super::{contextual_spec, finish_digest, ContextualRankContext, ContextualRankProducer};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::rank_features::providers::helpers::ensure_unconfigured;
use crate::schema::FeatureId;
use crate::state::CnotGate;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

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
