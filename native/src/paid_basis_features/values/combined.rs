use super::super::helpers::{resource, unsupported};
use super::{exact_count, matrix_bit};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::features::FeatureValue;
use crate::packet::FeatureStatus;
use crate::paid_basis::{cancel_adjacent_block_gates, endpoint_sha256, PaidShearCandidate};
use crate::state::{CnotGate, MatrixState};
use std::collections::HashMap;

pub(super) fn assemble_combined_feature_value(
    state: &MatrixState,
    alphabet: &CnotAlphabet,
    candidates: &[PaidShearCandidate],
) -> Result<(FeatureValue, Vec<String>), (FeatureStatus, String)> {
    let n = state.dimension() as usize;
    let word_blocks = candidates
        .iter()
        .map(|candidate| candidate.certificate.gates.clone())
        .collect::<Vec<_>>();
    let compatibility = adjacent_compatibility_counts(&word_blocks);
    let reduced = cancel_adjacent_block_gates(&word_blocks);
    let start = MatrixState::identity(state.dimension(), alphabet.id())
        .map_err(|error| unsupported(error.to_string()))?;
    let mut endpoint = start.clone();
    for gate in &reduced.word {
        endpoint
            .apply_cnot(*gate)
            .map_err(|error| unsupported(error.to_string()))?;
    }
    let combined_certificate = WordCertificate {
        start,
        end: endpoint.clone(),
        physical_cost: u64::try_from(reduced.word.len())
            .map_err(|_| resource("paid_basis_combined_word_length_exceeds_u64"))?,
        gates: reduced.word.clone(),
        producer: "verified_paid_basis_context_sequence_v1".into(),
        lineage_id: "explicit_paid_basis_contexts".into(),
    };
    combined_certificate
        .verify(alphabet)
        .map_err(|error| unsupported(error.to_string()))?;
    let endpoint_digest = endpoint_sha256(&endpoint);

    let pair_count = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| resource("paid_basis_d14_size_overflow"))?;
    let mut pair_values = Vec::new();
    pair_values
        .try_reserve_exact(pair_count)
        .map_err(|_| resource("paid_basis_d14_allocation_failed"))?;
    pair_values.resize(pair_count, 0.0f32);
    let mut gate_counts = HashMap::<(u32, u32), usize>::new();
    for gate in &reduced.word {
        let count = gate_counts.entry((gate.control, gate.target)).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| resource("paid_basis_d14_gate_count_overflow"))?;
    }
    let compatibility_counts = compatibility.into_iter().collect::<HashMap<_, _>>();
    let cancellation_counts = reduced
        .boundary_cancellations
        .iter()
        .map(|cancellation| {
            (
                (cancellation.control, cancellation.target),
                cancellation.canceled_pairs,
            )
        })
        .collect::<HashMap<_, _>>();
    for control in 0..n {
        for target in 0..n {
            let offset = (control * n + target) * 4;
            let endpoint_bit = matrix_bit(&endpoint, target, control);
            pair_values[offset] = f32::from(endpoint_bit);
            pair_values[offset + 1] = exact_count(
                gate_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
            pair_values[offset + 2] = exact_count(
                compatibility_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
            pair_values[offset + 3] = exact_count(
                cancellation_counts
                    .get(&(control as u32, target as u32))
                    .copied()
                    .unwrap_or(0),
            )?;
        }
    }
    let mut provenance = vec![format!("endpoint_sha256={endpoint_digest}")];
    for (index, candidate) in candidates.iter().enumerate() {
        provenance.push(format!(
            "block_{index}_endpoint_sha256={}",
            candidate.endpoint_sha256
        ));
    }
    let d14 = FeatureValue {
        shape: vec![n, n, 4],
        values: pair_values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    };

    Ok((d14, provenance))
}

fn adjacent_compatibility_counts(blocks: &[Vec<CnotGate>]) -> HashMap<(u32, u32), usize> {
    let mut counts = HashMap::new();
    for adjacent in blocks.windows(2) {
        for (left, right) in adjacent[0].iter().rev().zip(&adjacent[1]) {
            if left != right {
                break;
            }
            let count = counts.entry((left.control, left.target)).or_insert(0usize);
            *count += 1;
        }
    }
    counts
}
