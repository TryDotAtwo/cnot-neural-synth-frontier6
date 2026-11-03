use super::super::{validate_policy, FeatureBuildError, FeaturePolicy};
use super::FeatureRegistry;
use crate::schema::FeatureSpec;
use crate::structural;

impl FeatureRegistry {
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
