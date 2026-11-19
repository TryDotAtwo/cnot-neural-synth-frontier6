//! Finite symmetry orbit construction and split-overlap validation.
use super::alphabet::alphabet;
use super::types::{FrozenDataError, FrozenPanel};
use crate::alphabet::CnotAlphabet;
use crate::state::MatrixState;
use std::collections::{HashMap, HashSet};

pub(super) fn panel_views(panel: &FrozenPanel) -> Result<HashSet<MatrixState>, FrozenDataError> {
    let mut out = HashSet::new();
    for r in panel.records() {
        out.extend(views(&r.state)?);
    }
    Ok(out)
}
pub(super) fn views(state: &MatrixState) -> Result<HashSet<MatrixState>, FrozenDataError> {
    let inv = inverse(state)?;
    let trans = transpose(state)?;
    let inv_trans = transpose(&inv)?;
    let base = [state.clone(), inv, trans, inv_trans];
    let n = state.dimension() as usize;
    let reverse: Vec<u32> = (0..n as u32).rev().collect();
    let mut out = HashSet::new();
    for matrix in base {
        out.insert(matrix.clone());
        out.insert(permute(&matrix, &reverse)?);
    }
    Ok(out)
}
fn inverse(state: &MatrixState) -> Result<MatrixState, FrozenDataError> {
    let n = state.dimension() as usize;
    let w = state.words_per_row();
    let mut left = state.packed_rows().to_vec();
    let ident = MatrixState::identity(state.dimension(), state.alphabet_id())
        .map_err(|e| FrozenDataError(e.to_string()))?;
    let mut right = ident.packed_rows().to_vec();
    for col in 0..n {
        let word = col / 64;
        let bit = 1u64 << (col % 64);
        let pivot = (col..n)
            .find(|r| left[*r * w + word] & bit != 0)
            .ok_or_else(|| FrozenDataError("singular matrix in inverse".into()))?;
        if pivot != col {
            for k in 0..w {
                left.swap(pivot * w + k, col * w + k);
                right.swap(pivot * w + k, col * w + k);
            }
        }
        for row in 0..n {
            if row != col && left[row * w + word] & bit != 0 {
                for k in 0..w {
                    left[row * w + k] ^= left[col * w + k];
                    right[row * w + k] ^= right[col * w + k];
                }
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), right)
        .map_err(|e| FrozenDataError(e.to_string()))
}
fn transpose(state: &MatrixState) -> Result<MatrixState, FrozenDataError> {
    let n = state.dimension() as usize;
    let w = state.words_per_row();
    let mut out = vec![0u64; n * w];
    for i in 0..n {
        for j in 0..n {
            if getbit(state.packed_rows(), w, i, j) {
                putbit(&mut out, w, j, i);
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), out)
        .map_err(|e| FrozenDataError(e.to_string()))
}
fn permute(state: &MatrixState, p: &[u32]) -> Result<MatrixState, FrozenDataError> {
    let n = state.dimension() as usize;
    if p.len() != n {
        return Err(FrozenDataError("bad permutation".into()));
    }
    let w = state.words_per_row();
    let mut out = vec![0u64; n * w];
    for i in 0..n {
        for j in 0..n {
            if getbit(state.packed_rows(), w, i, j) {
                putbit(&mut out, w, p[i] as usize, p[j] as usize);
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), out)
        .map_err(|e| FrozenDataError(e.to_string()))
}
fn getbit(a: &[u64], w: usize, i: usize, j: usize) -> bool {
    a[i * w + j / 64] & (1u64 << (j % 64)) != 0
}
fn putbit(a: &mut [u64], w: usize, i: usize, j: usize) {
    a[i * w + j / 64] |= 1u64 << (j % 64);
}
pub(super) fn validate_train(
    train: &FrozenPanel,
    forbidden: &HashSet<MatrixState>,
    alphabets: &mut HashMap<u32, CnotAlphabet>,
) -> Result<(), FrozenDataError> {
    for (i, r) in train.records().iter().enumerate() {
        if views(&r.state)?.iter().any(|s| forbidden.contains(s)) {
            return Err(FrozenDataError(format!(
                "train[{i}] overlaps held-out views"
            )));
        }
        let alpha = alphabet(alphabets, r.state.dimension())?;
        // With fewer holdout states than gates, compare directly. A single
        // primitive changes exactly one row by XOR with a different source row.
        if forbidden.len() < alpha.gates().len() {
            if forbidden.iter().any(|s| single_gate_neighbor(&r.state, s)) {
                return Err(FrozenDataError(format!(
                    "train[{i}] one-hop overlaps holdout"
                )));
            }
            continue;
        }
        for gate in alpha.gates() {
            let mut next = r.state.clone();
            next.apply_cnot(*gate)
                .map_err(|e| FrozenDataError(e.to_string()))?;
            if forbidden.contains(&next) {
                return Err(FrozenDataError(format!(
                    "train[{i}] one-hop overlaps holdout"
                )));
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod neighbor_tests {
    use super::*;
    #[test]
    fn direct_check_matches_every_gate_on_gl3_and_across_word_boundaries() {
        for n in [3, 65] {
            let a = CnotAlphabet::all_to_all_unit(n).unwrap();
            let id = MatrixState::identity(n, a.id()).unwrap();
            let mut states = vec![id];
            let mut cursor = 0;
            while cursor < states.len() {
                let root = states[cursor].clone();
                let mut neighbors = HashSet::new();
                for gate in a.gates() {
                    let mut next = root.clone();
                    next.apply_cnot(*gate).unwrap();
                    assert!(single_gate_neighbor(&root, &next));
                    if n == 3 && !states.contains(&next) {
                        states.push(next.clone());
                    }
                    neighbors.insert(next);
                }
                for s in &states {
                    assert_eq!(single_gate_neighbor(&root, s), neighbors.contains(s));
                }
                if n == 65 {
                    let mut two = root.clone();
                    two.apply_cnot(crate::CnotGate {
                        control: 64,
                        target: 0,
                    })
                    .unwrap();
                    two.apply_cnot(crate::CnotGate {
                        control: 0,
                        target: 64,
                    })
                    .unwrap();
                    assert!(!single_gate_neighbor(&root, &two));
                }
                cursor += 1;
            }
        }
    }
}

fn single_gate_neighbor(root: &MatrixState, other: &MatrixState) -> bool {
    if root.dimension() != other.dimension() || root.alphabet_id() != other.alphabet_id() {
        return false;
    }
    root.gate_to_rows(other.packed_rows()).is_some()
}

pub(super) fn disjoint(
    a: &HashSet<MatrixState>,
    b: &HashSet<MatrixState>,
    why: &str,
) -> Result<(), FrozenDataError> {
    if a.iter().any(|x| b.contains(x)) {
        Err(FrozenDataError(why.into()))
    } else {
        Ok(())
    }
}
