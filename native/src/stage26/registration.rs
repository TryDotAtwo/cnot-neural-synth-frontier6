use super::producer::{SharedStage26, Stage26Producer};
use super::{
    CACHE_CAPACITY, DEFAULT_MAX_CHECKER_PAIRS, DEFAULT_MAX_N, STAGE26_CERTIFICATE_SOURCE_SHA256,
    STAGE26_CHECKER_SOURCE_SHA256, STAGE26_IDS, STAGE26_PROOF_NOTES_SHA256,
};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Registers exact F03/F04/F05 stage-26 features using the source defaults.
pub fn register_stage26_features(registry: &mut FeatureRegistry) -> Result<(), FeatureBuildError> {
    register_stage26_features_with_budget(registry, DEFAULT_MAX_N, DEFAULT_MAX_CHECKER_PAIRS)
}

/// Registers F03/F04/F05 with explicit width and independent-checker budgets.
pub fn register_stage26_features_with_budget(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_checker_pairs: usize,
) -> Result<(), FeatureBuildError> {
    if max_n == 0 || max_checker_pairs == 0 {
        return Err(FeatureBuildError(
            "stage26 requires positive width and checker-pair budgets".into(),
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"stage26_checked_features_v1");
    digest.update(STAGE26_CERTIFICATE_SOURCE_SHA256.as_bytes());
    digest.update(STAGE26_CHECKER_SOURCE_SHA256.as_bytes());
    digest.update(STAGE26_PROOF_NOTES_SHA256.as_bytes());
    digest.update(max_n.to_le_bytes());
    digest.update((max_checker_pairs as u64).to_le_bytes());
    let version = format!(
        "stage26_checked_features_v1:{}",
        hex_digest(digest.finalize().as_slice())
    );
    let shared = Arc::new(SharedStage26 {
        max_n,
        max_checker_pairs,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    });
    for (family, ordinal, channels) in STAGE26_IDS {
        let id = FeatureId::new(family, ordinal).expect("stage26 IDs are in the manifest");
        let role = if ordinal == 4 {
            FeatureRole::Pair
        } else {
            FeatureRole::Global
        };
        let spec = feature_spec(id).with_provider_metadata(
            "checked_stage26_canonical_projection;global_lower_bound_not_optimum",
            &version,
            ["M", "stage26_source", "independent_certificate_checker"],
            "exact",
            "T2",
            std::iter::empty::<&str>(),
            false,
        );
        let spec = FeatureSpec {
            role,
            channels,
            ..spec
        };
        registry.register(Box::new(Stage26Producer {
            spec,
            shared: Arc::clone(&shared),
        }))?;
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
