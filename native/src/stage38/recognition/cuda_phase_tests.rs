//! Locate source witnesses for late enumeration/recovery budget priority.
use super::super::contraction::{count_histories, recover_histories};
use super::super::{Histories, Stage38Error};
use crate::{CnotAlphabet, MatrixState};
use num_bigint::Sign;
use std::collections::HashSet;

fn phase(state: &MatrixState, budget: usize) -> Result<(u8, usize), Stage38Error> {
    let n = state.dimension() as usize;
    let w = n.div_ceil(64);
    let initial: Vec<_> = (0..n)
        .map(|r| state.row_words(r as u32).unwrap().to_vec())
        .collect();
    let final_rows: Vec<_> = (0..n)
        .map(|r| {
            let mut v = vec![0; w];
            v[r / 64] = 1 << (r % 64);
            v
        })
        .collect();
    let changed: Vec<_> = (0..n).filter(|&r| initial[r] != final_rows[r]).collect();
    let mut chosen = None;
    let mut positive = 0;
    for &target in &changed {
        let endpoints: HashSet<_> = (0..n)
            .filter(|&r| r != target)
            .flat_map(|r| [initial[r].clone(), final_rows[r].clone()])
            .collect();
        let mut candidates: Vec<_> = endpoints
            .iter()
            .map(|v| {
                initial[target]
                    .iter()
                    .zip(v)
                    .map(|(a, b)| a ^ b)
                    .collect::<Vec<_>>()
            })
            .filter(|v| {
                v.iter().any(|&x| x != 0) && *v != initial[target] && *v != final_rows[target]
            })
            .collect();
        candidates.sort_by(|a, b| a.iter().rev().cmp(b.iter().rev()));
        candidates.dedup();
        for middle in candidates {
            let diff: Vec<_> = middle
                .iter()
                .zip(&final_rows[target])
                .map(|(a, b)| a ^ b)
                .collect();
            if !endpoints.contains(&diff) {
                continue;
            }
            let h: Histories = (0..n)
                .map(|r| {
                    if r == target {
                        vec![initial[r].clone(), middle.clone(), final_rows[r].clone()]
                    } else if initial[r] != final_rows[r] {
                        vec![initial[r].clone(), final_rows[r].clone()]
                    } else {
                        vec![initial[r].clone()]
                    }
                })
                .collect();
            match count_histories(&h, budget, &HashSet::new()) {
                Ok(a) if a.count.sign() == Sign::Plus => {
                    positive += 1;
                    if chosen.is_none() {
                        chosen = Some(h);
                    }
                }
                Ok(_) => {}
                Err(Stage38Error::ContractionBudgetExceeded(_)) => return Ok((1, positive)),
                Err(e) => return Err(e),
            }
        }
    }
    if let Some(h) = chosen {
        match recover_histories(&h, budget) {
            Ok(_) => {}
            Err(Stage38Error::ContractionBudgetExceeded(_)) => return Ok((2, positive)),
            Err(e) => return Err(e),
        }
    }
    Ok((0, positive))
}
#[test]
fn late_enumeration_budget_overrides_positive_template() {
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let s = MatrixState::from_packed_rows(3, a.id(), vec![3, 5, 1]).unwrap();
    assert_eq!(phase(&s, 16).unwrap(), (1, 1));
    assert!(matches!(
        crate::recognize_stage38(&s, &a, 16).unwrap(),
        crate::Stage38Result::ResourceLimit(_)
    ));
    assert!(matches!(
        crate::recognize_stage38(&s, &a, 100_000).unwrap(),
        crate::Stage38Result::Exact(_)
    ));
}

#[test]
fn recovery_budget_overrides_complete_positive_enumeration() {
    let a = CnotAlphabet::all_to_all_unit(8).unwrap();
    let s = MatrixState::from_packed_rows(8, a.id(), vec![17, 130, 128, 40, 88, 160, 104, 132])
        .unwrap();
    assert_eq!(phase(&s, 729).unwrap(), (2, 5));
    assert!(matches!(
        crate::recognize_stage38(&s, &a, 729).unwrap(),
        crate::Stage38Result::ResourceLimit(_)
    ));
    assert!(matches!(
        crate::recognize_stage38(&s, &a, 100_000).unwrap(),
        crate::Stage38Result::Exact(_)
    ));
}
