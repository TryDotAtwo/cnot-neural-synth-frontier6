use super::super::{FeatureBuildError, FeatureProducer};
use super::FeatureRegistry;
use crate::schema::{feature_spec, is_reserved_native_layout, FeatureRole};

impl FeatureRegistry {
    pub fn register(
        &mut self,
        producer: Box<dyn FeatureProducer>,
    ) -> Result<(), FeatureBuildError> {
        let spec = producer.spec().clone();
        if !is_reserved_native_layout(spec.id) {
            let canonical = feature_spec(spec.id);
            if spec.role != canonical.role || spec.channels != canonical.channels {
                return Err(FeatureBuildError(format!(
                    "{} provider differs from the canonical feature layout",
                    spec.id
                )));
            }
        }
        if spec.channels == 0 {
            return Err(FeatureBuildError(format!(
                "{} provider must declare at least one channel",
                spec.id
            )));
        }
        if spec.role == FeatureRole::Group {
            let unique_roles: std::collections::HashSet<_> = spec.wire_roles.iter().collect();
            if spec.wire_roles.is_empty()
                || spec.wire_roles.iter().any(|role| role.trim().is_empty())
                || unique_roles.len() != spec.wire_roles.len()
            {
                return Err(FeatureBuildError(format!(
                    "{} group provider must declare unique physical wire roles",
                    spec.id
                )));
            }
        } else if !spec.wire_roles.is_empty() {
            return Err(FeatureBuildError(format!(
                "{} non-group provider cannot declare wire roles",
                spec.id
            )));
        }
        if spec.producer_version.trim().is_empty() || spec.producer_version == "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} provider has no version",
                spec.id
            )));
        }
        if spec.requires_solver_assisted && producer.symmetry_contract().trim().is_empty() {
            return Err(FeatureBuildError(format!(
                "{} solver-assisted provider lacks a contract",
                spec.id
            )));
        }
        if self.providers.contains_key(&spec.id) {
            return Err(FeatureBuildError(format!(
                "{} provider already registered",
                spec.id
            )));
        }
        if self
            .specs
            .get(&spec.id)
            .is_some_and(|existing| existing.producer_version != "unconfigured")
        {
            return Err(FeatureBuildError(format!(
                "{} already has a built-in provider",
                spec.id
            )));
        }
        self.specs.insert(spec.id, spec.clone());
        self.providers.insert(spec.id, producer);
        Ok(())
    }
}
