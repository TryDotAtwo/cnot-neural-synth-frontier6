//! Finite symmetry orbit construction and split-overlap validation.
use super::{FrozenDataError, FrozenPanel, alphabet};
use crate::state::MatrixState;
use std::collections::HashSet;

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
) -> Result<(), FrozenDataError> {
    for (i, r) in train.records().iter().enumerate() {
        if views(&r.state)?.iter().any(|s| forbidden.contains(s)) {
            return Err(FrozenDataError(format!(
                "train[{i}] overlaps held-out views"
            )));
        }
        for gate in alphabet(&r.state)?.gates() {
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
