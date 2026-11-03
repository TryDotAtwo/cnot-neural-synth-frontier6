//! Checked workspace and evaluation budgets for exact template response.

pub(super) fn source_evaluation_estimate(
    choice_count: usize,
    point_count: usize,
    assignment_count: usize,
) -> Option<usize> {
    let pair_count = choice_count.checked_mul(choice_count.saturating_sub(1))? / 2;
    let pair_evaluations = pair_count.checked_mul(4)?;
    let point_rounding_evaluations = point_count
        .checked_add(2)?
        .checked_mul(choice_count.checked_mul(2)?.checked_add(2)?)?;
    1usize
        .checked_add(point_count)?
        .checked_add(pair_evaluations)?
        .checked_add(point_rounding_evaluations)?
        .checked_add(assignment_count)
}

const RATIONAL_ITEM_BYTES: usize = 128;

pub(super) fn template_response_workspace_bytes(
    dimension: u32,
    choice_count: usize,
) -> Option<usize> {
    let width = (dimension / 2) as usize;
    1usize
        .checked_shl(width as u32)?
        .checked_mul(choice_count.saturating_add(6))?
        .checked_mul(RATIONAL_ITEM_BYTES)
}
