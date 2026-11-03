use super::{validate_policy, FeatureBuildError, FeaturePolicy, FeatureProducer};
use crate::schema::{
    all_feature_specs, feature_spec, is_reserved_native_layout, FeatureId, FeatureRole,
    FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID, FEATURE_LAYOUT_ID,
};
use crate::structural;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

mod builtins;

pub struct FeatureRegistry {
    pub(super) specs: HashMap<FeatureId, FeatureSpec>,
    pub(super) providers: HashMap<FeatureId, Box<dyn FeatureProducer>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut specs = all_feature_specs();
        for spec in &mut specs {
            if let Some(metadata) = structural::metadata(spec.id) {
                spec.scope = metadata.scope.into();
                spec.producer_version = metadata.producer_version.into();
                spec.dependencies = metadata
                    .dependencies
                    .iter()
                    .map(|value| (*value).into())
                    .collect();
                spec.exactness = metadata.exactness.into();
                spec.cost_tier = metadata.cost_tier.into();
                spec.wire_roles.clear();
            }
        }
        Self {
            specs: specs.into_iter().map(|spec| (spec.id, spec)).collect(),
            providers: HashMap::new(),
        }
    }

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

    /// Returns specs in the canonical 103-ID order, including the actual
    /// producer metadata used by this registry.
    pub fn specs(&self) -> Vec<FeatureSpec> {
        (0..FEATURE_COUNT)
            .filter_map(FeatureId::from_index)
            .map(|id| {
                self.specs
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| feature_spec(id))
            })
            .collect()
    }

    /// Stable digest of every feature layout/provider contract in canonical
    /// manifest order. Provider configuration belongs in `producer_version`.
    pub fn contract_sha256(&self) -> String {
        let mut digest = Sha256::new();
        hash_contract_field(&mut digest, b"cnot103_feature_registry_contract_v1");
        hash_contract_field(&mut digest, FEATURE_ID_MANIFEST_ID.as_bytes());
        hash_contract_field(&mut digest, FEATURE_LAYOUT_ID.as_bytes());
        for spec in self.specs() {
            hash_contract_field(&mut digest, spec.id.to_string().as_bytes());
            hash_contract_field(&mut digest, feature_role_name(spec.role).as_bytes());
            hash_contract_field(&mut digest, &(spec.channels as u64).to_le_bytes());
            hash_contract_field(&mut digest, spec.scope.as_bytes());
            hash_contract_field(&mut digest, spec.producer_version.as_bytes());
            hash_contract_field(&mut digest, &(spec.dependencies.len() as u64).to_le_bytes());
            for dependency in &spec.dependencies {
                hash_contract_field(&mut digest, dependency.as_bytes());
            }
            hash_contract_field(&mut digest, spec.exactness.as_bytes());
            hash_contract_field(&mut digest, spec.cost_tier.as_bytes());
            hash_contract_field(&mut digest, &(spec.wire_roles.len() as u64).to_le_bytes());
            for role in &spec.wire_roles {
                hash_contract_field(&mut digest, role.as_bytes());
            }
            hash_contract_field(&mut digest, &[u8::from(spec.requires_solver_assisted)]);
            let symmetry = self
                .providers
                .get(&spec.id)
                .map(|provider| provider.symmetry_contract())
                .unwrap_or_else(|| {
                    if structural::is_builtin(spec.id) {
                        "exact_relabel"
                    } else {
                        "unavailable"
                    }
                });
            hash_contract_field(&mut digest, symmetry.as_bytes());
        }
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Returns the policy's selected feature contracts in policy order.
    ///
    /// A model must not be constructed for a selected feature that has no
    /// executable native provider. All selected contract failures are returned
    /// together so one missing provider does not hide the rest. Built-in
    /// structural features are computed by the shared structural engine and
    /// therefore count as implemented.
    pub fn selected_specs(
        &self,
        policy: &FeaturePolicy,
    ) -> Result<Vec<FeatureSpec>, FeatureBuildError> {
        validate_policy(policy)?;
        let mut selected = Vec::with_capacity(policy.enabled_ids.len());
        let mut failures = Vec::new();
        for id in &policy.enabled_ids {
            let Some(spec) = self.specs.get(id) else {
                failures.push(format!("{id}: unknown feature ID"));
                continue;
            };

            let mut reasons = Vec::new();
            if spec.requires_solver_assisted && !policy.solver_assisted {
                reasons.push("requires solver-assisted policy");
            }
            if policy.symmetry_contract == "exact_relabel"
                && self
                    .providers
                    .get(id)
                    .is_some_and(|provider| provider.symmetry_contract() != "exact_relabel")
            {
                reasons.push("provider symmetry contract mismatch");
            }
            if spec.producer_version.trim().is_empty()
                || spec.producer_version == "unconfigured"
                || (!self.providers.contains_key(id) && !structural::is_builtin(*id))
            {
                reasons.push("no executable native provider");
            }

            if reasons.is_empty() {
                selected.push(spec.clone());
            } else {
                failures.push(format!("{id}: {}", reasons.join(", ")));
            }
        }
        if failures.is_empty() {
            Ok(selected)
        } else {
            Err(FeatureBuildError(format!(
                "feature policy is not executable: {}",
                failures.join("; ")
            )))
        }
    }
}

fn feature_role_name(role: FeatureRole) -> &'static str {
    match role {
        FeatureRole::Node => "node",
        FeatureRole::Pair => "pair",
        FeatureRole::Global => "global",
        FeatureRole::Group => "group",
    }
}

fn hash_contract_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}
