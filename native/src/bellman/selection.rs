mod branch;
mod candidates;
mod scoring;

use super::{BellmanError, BellmanSelection, MacroProvider};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::FusionDistanceField;
use crate::state::MatrixState;
use tch::nn::VarStore;

/// Select the minimum paid-cost primitive successor for every state.
pub fn select_backup(
    model: &FusionDistanceField,
    variables: &VarStore,
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    chunk_size: Option<usize>,
) -> Result<BellmanSelection, BellmanError> {
    select_backup_with_macros(
        model, variables, states, registry, policy, alphabet, None, chunk_size,
    )
}

/// Select the minimum paid-cost successor across all primitive and verified macro actions.
pub fn select_backup_with_macros(
    model: &FusionDistanceField,
    variables: &VarStore,
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    macro_provider: Option<&dyn MacroProvider>,
    chunk_size: Option<usize>,
) -> Result<BellmanSelection, BellmanError> {
    select_backup_internal(
        model,
        variables,
        states,
        registry,
        policy,
        alphabet,
        BellmanSelectionOptions {
            macro_provider,
            config: BellmanSelectionConfig {
                chunk_size,
                require_complete_model_input_coverage: false,
            },
        },
    )
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BellmanSelectionConfig {
    pub chunk_size: Option<usize>,
    pub require_complete_model_input_coverage: bool,
}

struct BellmanSelectionOptions<'a> {
    macro_provider: Option<&'a dyn MacroProvider>,
    config: BellmanSelectionConfig,
}

pub(crate) fn select_backup_with_model_input_coverage(
    model: &FusionDistanceField,
    variables: &VarStore,
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    config: BellmanSelectionConfig,
) -> Result<BellmanSelection, BellmanError> {
    select_backup_internal(
        model,
        variables,
        states,
        registry,
        policy,
        alphabet,
        BellmanSelectionOptions {
            macro_provider: None,
            config,
        },
    )
}

fn select_backup_internal(
    model: &FusionDistanceField,
    variables: &VarStore,
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    options: BellmanSelectionOptions<'_>,
) -> Result<BellmanSelection, BellmanError> {
    if states.is_empty() {
        return Err(BellmanError::Invalid("empty Bellman batch".into()));
    }
    if options.config.chunk_size.is_some_and(|size| size == 0) {
        return Err(BellmanError::Invalid(
            "positive neighbor chunk size required".into(),
        ));
    }
    if policy.evaluations.is_some() {
        return Err(BellmanError::Invalid(
            "finite feature-evaluation budgets can change the selected packets; use frozen unlimited deterministic producers".into(),
        ));
    }
    if variables.device() != model.device() {
        return Err(BellmanError::Invalid(
            "model and variable store are on different devices".into(),
        ));
    }

    let theta_digest = super::fingerprint::theta_fingerprint(variables)?;
    let mut branches = Vec::with_capacity(states.len());
    for state in states {
        branches.push(branch::prepare_branch(
            model, state, registry, policy, alphabet, &options,
        )?);
    }

    if super::fingerprint::theta_fingerprint(variables)? != theta_digest {
        return Err(BellmanError::Invalid(
            "model parameters changed during Bellman selection".into(),
        ));
    }
    Ok(BellmanSelection {
        branches,
        theta_digest,
    })
}

fn candidate_total(value: f64, cost: u64) -> Result<f64, BellmanError> {
    let total = value + cost as f64;
    if total.is_finite() {
        Ok(total)
    } else {
        Err(BellmanError::Invalid(
            "nonfinite Bellman candidate cost".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn candidate_cost_order_is_not_collapsed_to_float32_precision() {
        let lower = super::candidate_total(0.0, 16_777_216).unwrap();
        let higher = super::candidate_total(0.0, 16_777_217).unwrap();
        assert!(lower < higher);
        assert_eq!(lower as f32, higher as f32);
    }
}
