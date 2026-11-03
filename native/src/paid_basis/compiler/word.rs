use super::super::BoundaryCancellation;
use crate::state::CnotGate;
use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct ReducedWordAudit {
    pub(crate) word: Vec<CnotGate>,
    pub(crate) boundary_cancellations: Vec<BoundaryCancellation>,
}

pub(crate) fn cancel_adjacent_block_gates(blocks: &[Vec<CnotGate>]) -> ReducedWordAudit {
    let mut stack = Vec::<(CnotGate, usize)>::new();
    let mut boundary_counts = HashMap::<(u32, u32), usize>::new();
    for (block_index, block) in blocks.iter().enumerate() {
        for gate in block.iter().copied() {
            if stack.last().is_some_and(|(previous, _)| *previous == gate) {
                let (_, previous_block) = stack.pop().expect("the prior matching gate exists");
                if previous_block != block_index {
                    *boundary_counts
                        .entry((gate.control, gate.target))
                        .or_default() += 1;
                }
            } else {
                stack.push((gate, block_index));
            }
        }
    }
    let mut boundary_cancellations = boundary_counts
        .into_iter()
        .map(|((control, target), canceled_pairs)| BoundaryCancellation {
            control,
            target,
            canceled_pairs,
        })
        .collect::<Vec<_>>();
    boundary_cancellations.sort_by_key(|item| (item.control, item.target));
    ReducedWordAudit {
        word: stack.into_iter().map(|(gate, _)| gate).collect(),
        boundary_cancellations,
    }
}
