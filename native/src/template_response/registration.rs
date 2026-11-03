//! Registry wiring and stable configuration fingerprint for the response family.

use super::provider::{TemplateResponseCache, TemplateResponseProducer};
use super::{
    TemplateChoiceKind, TemplateResponseConfig, TEMPLATE_RESPONSE_IDS, TEMPLATE_RESPONSE_WIRE_ROLES,
};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Registers the source-defined E02/E03/E06-E09 finite-template feature family.
/// All six providers share one bounded per-state response cache.
pub fn register_template_response(
    registry: &mut FeatureRegistry,
    config: TemplateResponseConfig,
) -> Result<(), FeatureBuildError> {
    for ordinal in TEMPLATE_RESPONSE_IDS {
        let id = FeatureId::new(b'E', ordinal).expect("template response ID is in the manifest");
        if registry.specs()[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let shared = Arc::new(TemplateResponseCache::new(config.clone()));
    let digest = config_sha256(&config);
    for ordinal in TEMPLATE_RESPONSE_IDS {
        let id = FeatureId::new(b'E', ordinal).expect("template response ID is in the manifest");
        let mut spec = feature_spec(id);
        spec.role = if ordinal == 6 {
            FeatureRole::Group
        } else {
            FeatureRole::Global
        };
        spec.channels = match ordinal {
            3 => config.points.len(),
            8 => 2,
            9 => 5,
            _ => 1,
        };
        spec.scope =
            "selected_partial_shear;fixed_source_template;exact_finite_assignment_scope_only"
                .into();
        spec.producer_version = format!("finite_exact_template_response_e{ordinal:02}_v2:{digest}");
        spec.dependencies = vec![
            "M".into(),
            "deterministic_partial_shear".into(),
            "fixed_source_template_choices".into(),
            "rational_probe_points".into(),
            "exact_response_budgets".into(),
        ];
        spec.exactness = "float32_rounded_from_exact_rational".into();
        spec.cost_tier = "T2".into();
        spec.wire_roles = if ordinal == 6 {
            TEMPLATE_RESPONSE_WIRE_ROLES
                .iter()
                .map(|role| (*role).into())
                .collect()
        } else {
            Vec::new()
        };
        registry.register(Box::new(TemplateResponseProducer::new(
            id,
            spec,
            Arc::clone(&shared),
        )))?;
    }
    Ok(())
}

/// Backward-compatible entry point; registration now includes the complete
/// template-response family rather than E02 alone.
pub fn register_template_response_e02(
    registry: &mut FeatureRegistry,
    config: TemplateResponseConfig,
) -> Result<(), FeatureBuildError> {
    register_template_response(registry, config)
}

fn config_sha256(config: &TemplateResponseConfig) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"finite_exact_template_response_family_v2");
    hash_field(&mut digest, &(config.choices.len() as u64).to_le_bytes());
    for choice in &config.choices {
        hash_field(
            &mut digest,
            match choice.kind {
                TemplateChoiceKind::Add => b"add",
                TemplateChoiceKind::Swap => b"swap",
            },
        );
        hash_field(&mut digest, &(choice.control as u64).to_le_bytes());
        hash_field(&mut digest, &(choice.target as u64).to_le_bytes());
    }
    hash_field(&mut digest, &(config.points.len() as u64).to_le_bytes());
    for point in &config.points {
        hash_field(&mut digest, &(point.len() as u64).to_le_bytes());
        for probability in point {
            hash_field(&mut digest, &probability.numerator.to_le_bytes());
            hash_field(&mut digest, &probability.denominator.to_le_bytes());
        }
    }
    hash_field(
        &mut digest,
        &(config.budgets.max_width as u64).to_le_bytes(),
    );
    hash_field(
        &mut digest,
        &(config.budgets.max_assignments as u64).to_le_bytes(),
    );
    hash_field(
        &mut digest,
        &(config.budgets.max_evaluations as u64).to_le_bytes(),
    );
    hash_field(
        &mut digest,
        &(config.budgets.max_bytes as u64).to_le_bytes(),
    );
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}
