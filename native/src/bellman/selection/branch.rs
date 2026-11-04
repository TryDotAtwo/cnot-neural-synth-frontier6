use super::super::{BellmanBranchAudit, BellmanError, PreparedBranch};
use super::{candidates, scoring, BellmanSelectionConfig};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::model::FusionDistanceField;
use crate::state::MatrixState;

pub(super) fn prepare_branch(
    model: &FusionDistanceField,
    state: &MatrixState,
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    options: &BellmanSelectionConfig<'_>,
) -> Result<PreparedBranch, BellmanError> {
    if state.alphabet_id() != alphabet.id() || state.dimension() != alphabet.dimension() {
        return Err(BellmanError::Invalid(
            "state/alphabet mismatch in Bellman batch".into(),
        ));
    }
    let parent_packet = registry.build(std::slice::from_ref(state), policy)?;
    if options.require_complete_model_input_coverage {
        parent_packet
            .require_complete_model_input_coverage()
            .map_err(|error| BellmanError::Invalid(error.to_string()))?;
    }
    if state.is_identity() {
        let packet_digest = super::super::fingerprint::packet_fingerprint(&parent_packet, None);
        return Ok(PreparedBranch {
            parent: state.clone(),
            audit: BellmanBranchAudit {
                parent: parent_packet.state_keys[0].clone(),
                selected_index: None,
                selected_cost: None,
                candidate_count: 0,
                primitive_count: 0,
                packet_sha256: super::super::fingerprint::hex_digest(&packet_digest),
                macro_margin: None,
                primitive_totals: Vec::new(),
            },
            parent_packet,
            successor: None,
            successor_packet: None,
            cost: 0,
            packet_digest,
        });
    }

    let candidates = candidates::generate_candidates(state, alphabet, options.macro_provider)?;
    let primitive_count = candidates.primitive_count;
    let scored = scoring::score_candidates(model, &candidates, registry, policy, *options)?;
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
        super::super::fingerprint::packet_fingerprint(&parent_packet, Some(&selected_packet));
    Ok(PreparedBranch {
        parent: state.clone(),
        audit: BellmanBranchAudit {
            parent: parent_packet.state_keys[0].clone(),
            selected_index: Some(selected_index),
            selected_cost,
            candidate_count: candidates.successors.len(),
            primitive_count,
            packet_sha256: super::super::fingerprint::hex_digest(&packet_digest),
            macro_margin,
            primitive_totals: totals[..primitive_count].to_vec(),
        },
        parent_packet,
        successor: Some(successor),
        successor_packet: Some(selected_packet),
        cost: selected_cost.unwrap_or_default(),
        packet_digest,
    })
}
