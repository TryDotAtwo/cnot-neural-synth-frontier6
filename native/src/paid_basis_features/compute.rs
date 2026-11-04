//! Build verified exact-local candidates before deriving their feature values.
use super::cache::CachedPaidBasisFeatures;
use super::helpers::{status_for_compile_error, workspace_estimate};
use super::values::{assemble_feature_values, matrix_bit};
use super::PaidBasisBlockContext;
use crate::alphabet::CnotAlphabet;
use crate::features::FeaturePolicy;
use crate::packet::FeatureStatus;
use crate::paid_basis::{compile_paid_shear, PaidShearConfig};
use crate::state::MatrixState;

pub(super) fn compute_paid_basis_features(
    state: &MatrixState,
    policy: &FeaturePolicy,
    all_contexts: &[PaidBasisBlockContext],
    config: PaidShearConfig,
) -> CachedPaidBasisFeatures {
    let fail = |status, reason: String| CachedPaidBasisFeatures::Unavailable(status, reason);
    let dimension = state.dimension();
    let relevant = all_contexts
        .iter()
        .filter(|context| context.dimension == dimension)
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return fail(
            FeatureStatus::Unsupported,
            "no_paid_basis_context_for_matrix_dimension".into(),
        );
    }
    let alphabet = match CnotAlphabet::all_to_all_unit(dimension) {
        Ok(alphabet) if alphabet.id() == state.alphabet_id() => alphabet,
        _ => {
            return fail(
                FeatureStatus::Unsupported,
                "paid_basis_requires_all_to_all_unit_CNOT_alphabet".into(),
            );
        }
    };
    let Some(estimated_bytes) = workspace_estimate(dimension as usize, relevant.len()) else {
        return fail(
            FeatureStatus::ResourceLimit,
            "paid_basis_workspace_size_overflow".into(),
        );
    };
    if policy
        .max_bytes
        .is_some_and(|limit| estimated_bytes > limit)
    {
        return fail(
            FeatureStatus::ResourceLimit,
            "paid_basis_workspace_budget_exceeded".into(),
        );
    }

    let mut candidates = Vec::new();
    if candidates.try_reserve_exact(relevant.len()).is_err() {
        return fail(
            FeatureStatus::ResourceLimit,
            "paid_basis_candidate_allocation_failed".into(),
        );
    }
    for context in &relevant {
        let block = match extract_block(state, context) {
            Ok(block) => block,
            Err(reason) => return fail(FeatureStatus::ResourceLimit, reason),
        };
        match compile_paid_shear(
            &block,
            dimension as usize,
            &context.source_wires,
            &context.target_wires,
            config,
        ) {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => return fail(status_for_compile_error(&error), error.to_string()),
        }
    }

    match assemble_feature_values(state, &alphabet, &relevant, &candidates) {
        Ok(values) => CachedPaidBasisFeatures::Computed(Box::new(values)),
        Err((status, reason)) => fail(status, reason),
    }
}

fn extract_block(
    state: &MatrixState,
    context: &PaidBasisBlockContext,
) -> Result<Vec<Vec<u8>>, String> {
    let cells = context
        .source_wires
        .len()
        .checked_mul(context.target_wires.len())
        .ok_or_else(|| "paid_basis_block_size_overflow".to_owned())?;
    let mut block = Vec::new();
    block
        .try_reserve_exact(context.target_wires.len())
        .map_err(|_| "paid_basis_block_allocation_failed".to_owned())?;
    for &target in &context.target_wires {
        let mut row = Vec::new();
        row.try_reserve_exact(context.source_wires.len())
            .map_err(|_| "paid_basis_block_allocation_failed".to_owned())?;
        for &source in &context.source_wires {
            row.push(matrix_bit(state, target, source));
        }
        block.push(row);
    }
    if block.iter().map(Vec::len).sum::<usize>() != cells {
        return Err("paid_basis_block_extraction_mismatch".into());
    }
    Ok(block)
}
