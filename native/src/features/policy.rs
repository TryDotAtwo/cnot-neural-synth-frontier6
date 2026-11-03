use super::FeatureBuildError;
use crate::schema::{all_feature_specs, all_model_input_specs, FeatureId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePolicy {
    pub enabled_ids: Vec<FeatureId>,
    pub seed: u64,
    pub schema_id: String,
    pub solver_assisted: bool,
    pub symmetry_contract: String,
    pub evaluations: Option<usize>,
    pub max_bytes: Option<usize>,
}

impl FeaturePolicy {
    pub fn complete_103(schema_id: impl Into<String>) -> Self {
        Self {
            enabled_ids: all_feature_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect(),
            seed: 0,
            schema_id: schema_id.into(),
            solver_assisted: false,
            symmetry_contract: "exact_relabel".into(),
            evaluations: None,
            max_bytes: None,
        }
    }

    /// Selects the canonical 103-ID manifest with the weaker symmetry
    /// contract needed by seeded Monte Carlo and physical-wire probes.
    pub fn complete_103_augmentation_only(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::complete_103(schema_id);
        policy.symmetry_contract = "augmentation_only".into();
        policy
    }

    /// Select the executable model inputs from the canonical 103-ID manifest.
    /// H07 remains a post-forward diagnostic, not a model input.
    pub fn all_model_inputs(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::complete_103(schema_id);
        policy.enabled_ids = all_model_input_specs()
            .into_iter()
            .map(|spec| spec.id)
            .collect();
        policy
    }

    /// All model inputs with the weaker symmetry contract needed by seeded
    /// Monte Carlo and physical-wire probes.
    pub fn all_model_inputs_augmentation_only(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::all_model_inputs(schema_id);
        policy.symmetry_contract = "augmentation_only".into();
        policy
    }

    pub fn id(&self) -> Result<String, FeatureBuildError> {
        validate_policy(self)?;
        // Stable, length-delimited textual form suitable for logs and checkpoint binding.
        // The campaign layer adds a cryptographic digest when it commits artifacts.
        let mut body = String::new();
        push_field(&mut body, &self.schema_id);
        push_field(&mut body, &self.seed.to_string());
        push_field(
            &mut body,
            if self.solver_assisted {
                "solver=1"
            } else {
                "solver=0"
            },
        );
        push_field(&mut body, &self.symmetry_contract);
        push_field(
            &mut body,
            &self
                .evaluations
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        push_field(
            &mut body,
            &self
                .max_bytes
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        for id in &self.enabled_ids {
            push_field(&mut body, &id.to_string());
        }
        Ok(body)
    }
}

fn push_field(output: &mut String, value: &str) {
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}

pub(super) fn validate_policy(policy: &FeaturePolicy) -> Result<(), FeatureBuildError> {
    if policy.enabled_ids.is_empty() {
        return Err(FeatureBuildError(
            "feature policy selects no feature IDs".into(),
        ));
    }
    if policy.schema_id.trim().is_empty() || policy.symmetry_contract.trim().is_empty() {
        return Err(FeatureBuildError(
            "feature policy needs schema and symmetry identities".into(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(policy.enabled_ids.len());
    if policy.enabled_ids.iter().any(|id| !seen.insert(*id)) {
        return Err(FeatureBuildError(
            "feature policy repeats a feature ID".into(),
        ));
    }
    Ok(())
}
