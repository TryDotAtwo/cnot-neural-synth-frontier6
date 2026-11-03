use super::{AuxiliaryHeads, FamilyHeads, FusionDistanceField, FusionTransformer, StructuralStem};
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::{ModelConfig, ModelError};
use crate::schema::{FeatureRole, FeatureSpec};
use tch::nn;
use tch::Device;

impl FusionDistanceField {
    /// Constructs a model directly from the registry and feature policy used
    /// to build its packets, so the model contract cannot drift from either.
    pub fn new_for_policy(
        path: &nn::Path<'_>,
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        let specs = registry
            .selected_specs(policy)
            .map_err(|error| ModelError(error.0))?;
        let policy_id = policy.id().map_err(|error| ModelError(error.0))?;
        Self::new(
            path,
            &specs,
            policy.schema_id.clone(),
            policy_id,
            device,
            config,
        )
    }

    pub fn new(
        path: &nn::Path<'_>,
        specs: &[FeatureSpec],
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        crate::torch_device::require_device(device).map_err(ModelError)?;
        config.validate()?;
        let schema_id = schema_id.into();
        let policy_id = policy_id.into();
        if schema_id.trim().is_empty() || policy_id.trim().is_empty() {
            return Err(ModelError(
                "model requires explicit schema and policy identities".into(),
            ));
        }
        if specs.is_empty() {
            return Err(ModelError(
                "model requires at least one selected feature spec".into(),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(specs.len());
        let mut families: Vec<_> = specs.iter().map(|spec| spec.id.family()).collect();
        for spec in specs {
            if !seen.insert(spec.id) {
                return Err(ModelError(format!(
                    "{} appears more than once in the model feature schema",
                    spec.id
                )));
            }
            if spec.channels == 0 || i64::try_from(spec.channels).is_err() {
                return Err(ModelError(format!(
                    "{} has an invalid channel count",
                    spec.id
                )));
            }
            if spec.producer_version.trim().is_empty()
                || spec.producer_version == "unconfigured"
                || spec.scope.trim().is_empty()
                || spec.exactness.trim().is_empty()
                || spec.exactness == "unknown"
                || spec.cost_tier.trim().is_empty()
                || spec
                    .dependencies
                    .iter()
                    .any(|value| value.trim().is_empty())
            {
                return Err(ModelError(format!(
                    "{} does not have an accepted producer contract",
                    spec.id
                )));
            }
            if spec.role == FeatureRole::Group
                && (spec.wire_roles.is_empty()
                    || spec.wire_roles.iter().any(|role| role.trim().is_empty())
                    || spec
                        .wire_roles
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != spec.wire_roles.len())
            {
                return Err(ModelError(format!(
                    "{} requires a unique physical wire-role layout",
                    spec.id
                )));
            }
            if spec.role != FeatureRole::Group && !spec.wire_roles.is_empty() {
                return Err(ModelError(format!(
                    "{} has group roles on a non-group feature",
                    spec.id
                )));
            }
        }
        families.sort_unstable();
        families.dedup();

        Ok(Self {
            specs: specs.to_vec(),
            expected_schema_id: schema_id,
            expected_policy_id: policy_id,
            device,
            stem: StructuralStem::new(&(path / "stem"), specs, &config),
            family_heads: FamilyHeads::new(&(path / "family_heads"), &families, &config),
            fusion: FusionTransformer::new(&(path / "fusion"), &config, families.len()),
            auxiliary: AuxiliaryHeads::new(&(path / "auxiliary"), &config),
        })
    }
}
