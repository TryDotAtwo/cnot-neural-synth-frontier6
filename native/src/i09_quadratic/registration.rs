use super::producer::I09Producer;
use super::types::{context_digest, I09Context};
use super::{I09Limits, I09_STAGE60_SOURCE_SHA256, MAX_EXACT_F32_INTEGER};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Registers I09 for explicit width-labelled Stage-60 compiler contexts.
pub fn register_i09_quadratic_response(
    registry: &mut FeatureRegistry,
    contexts: Vec<I09Context>,
    limits: I09Limits,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty()
        || limits.max_variables == 0
        || limits.max_equations == 0
        || limits.max_bytes == 0
    {
        return Err(FeatureBuildError(
            "I09 requires explicit contexts and positive resource limits".into(),
        ));
    }
    let mut widths = BTreeSet::new();
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I09 permits one explicit context per matrix width".into(),
        ));
    }
    if limits.max_variables > MAX_EXACT_F32_INTEGER || limits.max_equations > MAX_EXACT_F32_INTEGER
    {
        return Err(FeatureBuildError(
            "I09 float labels must remain exact integers".into(),
        ));
    }
    let id = FeatureId::new(b'I', 9).expect("I09 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = limits
        .max_variables
        .checked_add(4)
        .ok_or_else(|| FeatureBuildError("I09 channel count overflows the platform size".into()))?;
    let version = producer_version(&contexts, limits);
    let spec = spec.with_provider_metadata(
        "pinned_stage60_optional_cnot_quadratic_equations;alpha,beta,rank_B,mixed_rank;not_solver_or_distance",
        version,
        ["M", "explicit_optional_cnot_context", "stage60_source_sha256"],
        "exact_within_declared_context_and_resource_limits",
        "T2",
        ["equation_physical_support"],
        false,
    );
    registry.register(Box::new(I09Producer {
        spec,
        contexts,
        limits,
    }))
}

fn producer_version(contexts: &[I09Context], limits: I09Limits) -> String {
    let mut digest = Sha256::new();
    digest.update(b"i09_stage60_quadratic_response_rust_v1");
    digest.update(I09_STAGE60_SOURCE_SHA256.as_bytes());
    digest.update((limits.max_variables as u64).to_le_bytes());
    digest.update((limits.max_equations as u64).to_le_bytes());
    digest.update((limits.max_bytes as u64).to_le_bytes());
    for context in contexts {
        digest.update(context_digest(context).as_bytes());
    }
    format!("i09_stage60_rust_v1:{:x}", digest.finalize())
}

/// Helper for consumers that need to assert the pinned source at runtime.
pub fn i09_source_sha256() -> &'static str {
    I09_STAGE60_SOURCE_SHA256
}
