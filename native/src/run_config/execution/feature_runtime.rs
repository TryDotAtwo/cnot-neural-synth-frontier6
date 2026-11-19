//! Construction of the feature-provider contract selected by a run manifest.
#[cfg(test)]
mod budget_tests;

use super::super::manifest::{FeatureContextsSpec, FeaturePolicySpec};
use crate::alphabet::CnotAlphabet;
use crate::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan};
use crate::features::{FeaturePolicy, FeatureRegistry};
use std::collections::{BTreeSet, HashMap};

pub(super) struct FeatureRuntime {
    pub(super) registry: FeatureRegistry,
    pub(super) policy: FeaturePolicy,
    pub(super) selected_count: usize,
    pub(super) plans: HashMap<u32, DeviceFeaturePlan>,
}

pub(super) fn build(
    contexts_spec: FeatureContextsSpec,
    policy_spec: FeaturePolicySpec,
    alphabets: &[CnotAlphabet],
    seed: u64,
    widths: BTreeSet<u32>,
    index: usize,
) -> Result<FeatureRuntime, String> {
    let f08_max_n = contexts_spec.f08_max_n();
    let f08_max_changed_rows = contexts_spec.f08_max_changed_rows();
    let f08_max_component_events = contexts_spec.f08_max_component_events();
    let f08_max_history_counts = contexts_spec.f08_max_history_counts();
    let contexts = contexts_spec.build()?;
    let profile_alphabets = std::iter::once(&contexts.alphabet).chain(
        contexts.selected_residual_additional_contexts.iter().map(|context| &context.alphabet),
    );
    for profile_alphabet in profile_alphabets {
        if !alphabets.iter().any(|alphabet| {
            alphabet.id() == profile_alphabet.id()
                && alphabet.dimension() == profile_alphabet.dimension()
                && alphabet.gates() == profile_alphabet.gates()
                && (0..alphabet.gates().len())
                    .all(|i| alphabet.cost_at(i) == profile_alphabet.cost_at(i))
        }) {
            return Err("feature-context alphabet must exactly match an objective alphabet".into());
        }
    }

    let registry =
        FeatureRegistry::with_all_model_input_providers_f08_limit(contexts.clone(), f08_max_n, f08_max_changed_rows, f08_max_component_events, f08_max_history_counts)
            .map_err(|e| e.to_string())?;
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

    let mut plans = HashMap::new();
    for n in widths {
        let plan = DeviceFeaturePlan::with_contexts_f08_limit(
            &registry,
            &policy,
            n,
            index,
            DeviceFeatureContextInputs::from(&contexts),
            Some(f08_max_n),
            f08_max_changed_rows,
            f08_max_component_events,
            f08_max_history_counts,
        )?;
        plans.insert(n, plan);
    }
    Ok(FeatureRuntime {
        registry,
        policy,
        selected_count,
        plans,
    })
}
