//! Construction of the feature-provider contract selected by a run manifest.

use super::super::manifest::{FeatureContextsSpec, FeaturePolicySpec};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeaturePolicy, FeatureRegistry};

pub(super) struct FeatureRuntime {
    pub(super) registry: FeatureRegistry,
    pub(super) policy: FeaturePolicy,
    pub(super) selected_count: usize,
}

pub(super) fn build(
    contexts_spec: FeatureContextsSpec,
    policy_spec: FeaturePolicySpec,
    alphabets: &[CnotAlphabet],
    seed: u64,
) -> Result<FeatureRuntime, String> {
    let contexts = contexts_spec.build()?;
    let profile_alphabet = &contexts.alphabet;
    if !alphabets.iter().any(|alphabet| {
        alphabet.id() == profile_alphabet.id()
            && alphabet.dimension() == profile_alphabet.dimension()
            && alphabet.gates() == profile_alphabet.gates()
            && (0..alphabet.gates().len())
                .all(|i| alphabet.cost_at(i) == profile_alphabet.cost_at(i))
    }) {
        return Err("feature-context alphabet must exactly match an objective alphabet".into());
    }

    let registry =
        FeatureRegistry::with_all_model_input_providers(contexts).map_err(|e| e.to_string())?;
    let mut policy = FeaturePolicy::all_model_inputs_augmentation_only(policy_spec.schema_id);
    policy.seed = seed;
    policy.solver_assisted = policy_spec.solver_assisted;
    policy.evaluations = policy_spec.evaluations.into_option();
    policy.max_bytes = policy_spec.max_bytes.into_option();
    let selected_count = registry
        .selected_specs(&policy)
        .map_err(|e| e.to_string())?
        .len();
    if selected_count != crate::schema::MODEL_INPUT_COUNT {
        return Err(format!(
            "expected all {} model inputs, registry selected {}",
            crate::schema::MODEL_INPUT_COUNT,
            selected_count
        ));
    }

    Ok(FeatureRuntime {
        registry,
        policy,
        selected_count,
    })
}
