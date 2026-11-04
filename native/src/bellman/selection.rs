mod candidates;
mod scoring;

use super::{BellmanBranchAudit, BellmanError, BellmanSelection, MacroProvider, PreparedBranch};
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
        if state.alphabet_id() != alphabet.id() || state.dimension() != alphabet.dimension() {
            return Err(BellmanError::Invalid(
                "state/alphabet mismatch in Bellman batch".into(),
            ));
        }
        let parent_packet = registry.build(std::slice::from_ref(state), policy)?;
        if options.config.require_complete_model_input_coverage {
            parent_packet
                .require_complete_model_input_coverage()
                .map_err(|error| BellmanError::Invalid(error.to_string()))?;
        }
        if state.is_identity() {
            let packet_digest = super::fingerprint::packet_fingerprint(&parent_packet, None);
            branches.push(PreparedBranch {
                parent: state.clone(),
                audit: BellmanBranchAudit {
                    parent: parent_packet.state_keys[0].clone(),
                    selected_index: None,
                    selected_cost: None,
                    candidate_count: 0,
                    primitive_count: 0,
                    packet_sha256: super::fingerprint::hex_digest(&packet_digest),
                    macro_margin: None,
                    primitive_totals: Vec::new(),
                },
                parent_packet,
                successor: None,
                successor_packet: None,
                cost: 0,
                packet_digest,
            });
            continue;
        }

        let candidates = candidates::generate_candidates(state, alphabet, options.macro_provider)?;
        let primitive_count = candidates.primitive_count;
        let scored =
            scoring::score_candidates(model, &candidates, registry, policy, options.config)?;
        let selected_index = scored.selected_index;
        let selected_cost = scored.selected_cost;
        let selected_packet = scored.selected_packet;
        let totals = scored.totals;
        let selected_index = selected_index
            .ok_or_else(|| BellmanError::Invalid("no finite Bellman candidate selected".into()))?;
        let min_primitive = totals[..primitive_count].iter().copied().reduce(f64::min);
        let min_macro = totals[primitive_count..].iter().copied().reduce(f64::min);
        let macro_margin = min_macro
            .zip(min_primitive)
            .map(|(macro_cost, primitive_cost)| macro_cost - primitive_cost);
        let successor = candidates.successors[selected_index].clone();
        let selected_packet = selected_packet
            .ok_or_else(|| BellmanError::Invalid("selected successor packet is missing".into()))?;
        let packet_digest =
            super::fingerprint::packet_fingerprint(&parent_packet, Some(&selected_packet));
        branches.push(PreparedBranch {
            parent: state.clone(),
            audit: BellmanBranchAudit {
                parent: parent_packet.state_keys[0].clone(),
                selected_index: Some(selected_index),
                selected_cost,
                candidate_count: candidates.successors.len(),
                primitive_count,
                packet_sha256: super::fingerprint::hex_digest(&packet_digest),
                macro_margin,
                primitive_totals: totals[..primitive_count].to_vec(),
            },
            parent_packet,
            successor: Some(successor),
            successor_packet: Some(selected_packet),
            cost: selected_cost.unwrap_or_default(),
            packet_digest,
        });
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
