use super::Stage35Error;
use crate::alphabet::CnotAlphabet;
use crate::state::{CnotGate, MatrixState};

pub(super) const IDENTITY_FRAME: [u8; 2] = [1, 2];
pub(super) const FRAME_IDENTITY: &[(usize, usize)] = &[];
const FRAME_13: &[(usize, usize)] = &[(0, 1)];
const FRAME_32: &[(usize, usize)] = &[(1, 0)];
const FRAME_23: &[(usize, usize)] = &[(0, 1), (1, 0)];
const FRAME_31: &[(usize, usize)] = &[(1, 0), (0, 1)];
const FRAME_21: &[(usize, usize)] = &[(0, 1), (1, 0), (0, 1)];
pub(super) const FRAME_IDENTITY_WITH_SHARED: &[(usize, usize)] = &[(0, 1), (0, 1)];

pub(super) fn is_full_unit_alphabet(alphabet: &CnotAlphabet) -> Result<bool, Stage35Error> {
    let n = usize::try_from(alphabet.dimension()).map_err(|_| Stage35Error::IndexOverflow)?;
    let Some(action_count) = n.checked_mul(n.saturating_sub(1)) else {
        return Ok(false);
    };
    if alphabet.gates().len() != action_count {
        return Ok(false);
    }
    if alphabet
        .gates()
        .iter()
        .enumerate()
        .any(|(index, _)| alphabet.cost_at(index) != Some(1))
    {
        return Ok(false);
    }
    Ok(true)
}

pub(super) fn is_identity_row(state: &MatrixState, row: usize) -> bool {
    state.row_words(row as u32).is_some_and(|words| {
        words.iter().enumerate().all(|(word, value)| {
            *value
                == if word == row / 64 {
                    1u64 << (row % 64)
                } else {
                    0
                }
        })
    })
}

pub(super) fn bit(state: &MatrixState, row: usize, column: usize) -> bool {
    state.row_words(row as u32).is_some_and(|words| {
        words
            .get(column / 64)
            .is_some_and(|word| word & (1u64 << (column % 64)) != 0)
    })
}

pub(super) fn restricted_row(state: &MatrixState, row: usize, active_rows: [usize; 2]) -> u8 {
    u8::from(bit(state, row, active_rows[0])) | (u8::from(bit(state, row, active_rows[1])) << 1)
}

pub(super) fn multiply_2x2(matrix_rows: [u8; 2], vector: u8) -> u8 {
    ((matrix_rows[0] & vector).count_ones() as u8 & 1)
        | (((matrix_rows[1] & vector).count_ones() as u8 & 1) << 1)
}

pub(super) fn inverse_2x2(rows: [u8; 2]) -> Result<[u8; 2], Stage35Error> {
    let a = rows[0] & 1;
    let x = (rows[0] >> 1) & 1;
    let y = rows[1] & 1;
    let d = (rows[1] >> 1) & 1;
    if (a & d) ^ (x & y) != 1 {
        return Err(Stage35Error::SingularActiveBlock);
    }
    Ok([d | (x << 1), y | (a << 1)])
}

pub(super) fn frame_for(rows: [u8; 2]) -> Option<&'static [(usize, usize)]> {
    match rows {
        [1, 2] => Some(FRAME_IDENTITY),
        [1, 3] => Some(FRAME_13),
        [3, 2] => Some(FRAME_32),
        [2, 3] => Some(FRAME_23),
        [3, 1] => Some(FRAME_31),
        [2, 1] => Some(FRAME_21),
        _ => None,
    }
}

pub(super) fn gate(control: usize, target: usize) -> Result<CnotGate, Stage35Error> {
    Ok(CnotGate {
        control: u32::try_from(control).map_err(|_| Stage35Error::IndexOverflow)?,
        target: u32::try_from(target).map_err(|_| Stage35Error::IndexOverflow)?,
    })
}

pub(super) fn support_components(state: &MatrixState) -> usize {
    let n = state.dimension() as usize;
    let mut parent: Vec<usize> = (0..n).collect();
    for row in 0..n {
        for column in 0..n {
            if row != column && bit(state, row, column) {
                union(&mut parent, row, column);
            }
        }
    }
    let mut roots = std::collections::HashSet::with_capacity(n);
    for row in 0..n {
        roots.insert(find(&mut parent, row));
    }
    roots.len()
}

fn find(parent: &mut [usize], mut value: usize) -> usize {
    while parent[value] != value {
        let grandparent = parent[parent[value]];
        parent[value] = grandparent;
        value = grandparent;
    }
    value
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        parent[left_root] = right_root;
    }
}
