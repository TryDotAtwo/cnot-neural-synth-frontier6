//! Exact cost evaluation for an ordered template-choice assignment.

use super::{TemplateChoice, TemplateChoiceKind};

pub(super) fn template_assignment_cost(
    block: &[Vec<u8>],
    choices: &[TemplateChoice],
    assignment: usize,
) -> Option<u64> {
    let mut transformed = block.to_vec();
    let mut preparation_cost = 0u64;
    for (index, choice) in choices.iter().copied().enumerate() {
        if assignment & (1usize << index) == 0 {
            continue;
        }
        match choice.kind {
            TemplateChoiceKind::Add => {
                preparation_cost = preparation_cost.checked_add(2)?;
                for row in &mut transformed {
                    row[choice.control] ^= row[choice.target];
                }
            }
            TemplateChoiceKind::Swap => {
                preparation_cost = preparation_cost.checked_add(6)?;
                for row in &mut transformed {
                    row.swap(choice.control, choice.target);
                }
            }
        }
    }
    let residual_cost = transformed
        .iter()
        .flatten()
        .try_fold(0u64, |sum, bit| sum.checked_add(u64::from(*bit)))?;
    preparation_cost.checked_add(residual_cost)
}
