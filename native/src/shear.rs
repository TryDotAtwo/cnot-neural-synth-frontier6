//! Exact full-shear recognition and deterministic partial elimination.
//!
//! This module has no Torch dependency so feature producers and the paid-basis
//! adapter use the same source-bound decomposition implementation.

use crate::state::MatrixState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisBinding {
    FullShear,
    PartialElimination,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShearDecomposition {
    pub source_wires: Vec<usize>,
    pub target_wires: Vec<usize>,
    pub block: Vec<Vec<u8>>,
    pub pivot_columns: Option<Vec<usize>>,
}

pub fn recognize_shear(state: &MatrixState) -> Option<ShearDecomposition> {
    let n = state.dimension() as usize;
    let words_per_row = state.words_per_row();
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    let mut source_mask = vec![0u64; words_per_row];

    for row in 0..n {
        let packed = state.row_words(row as u32)?;
        let is_identity = (0..words_per_row).all(|word| {
            let expected = if word == row / 64 {
                1u64 << (row % 64)
            } else {
                0
            };
            packed[word] == expected
        });
        if is_identity {
            sources.push(row);
            source_mask[row / 64] |= 1u64 << (row % 64);
        } else {
            targets.push(row);
        }
    }
    if sources.is_empty() || targets.is_empty() {
        return None;
    }

    let mut block: Vec<Vec<u8>> = Vec::with_capacity(targets.len());
    for &target in &targets {
        let row = state.row_words(target as u32)?;
        for word in 0..words_per_row {
            let diagonal = if word == target / 64 {
                1u64 << (target % 64)
            } else {
                0
            };
            if (row[word] ^ diagonal) & !source_mask[word] != 0 {
                return None;
            }
        }
        block.push(
            sources
                .iter()
                .map(|&source| ((row[source / 64] >> (source % 64)) & 1) as u8)
                .collect(),
        );
    }

    Some(ShearDecomposition {
        source_wires: sources,
        target_wires: targets,
        block,
        pivot_columns: None,
    })
}

pub fn partial_shear(state: &MatrixState) -> Result<Option<ShearDecomposition>, String> {
    let n = state.dimension() as usize;
    let width = n / 2;
    if width == 0 {
        return Ok(None);
    }
    let sources = (0..width).collect::<Vec<_>>();
    let targets = (width..n).collect::<Vec<_>>();
    let words_per_row = state.words_per_row();
    let mut echelon = (0..width)
        .map(|row| state.row_words(row as u32).map(<[u64]>::to_vec))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "partial-shear source row is absent".to_owned())?;
    let mut pivot_columns = Vec::with_capacity(width);
    let mut pivot_row = 0usize;

    for column in 0..n {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(found) = (pivot_row..width).find(|&row| echelon[row][word] & bit != 0) else {
            continue;
        };
        echelon.swap(pivot_row, found);
        for row in (pivot_row + 1)..width {
            if echelon[row][word] & bit != 0 {
                for index in 0..words_per_row {
                    echelon[row][index] ^= echelon[pivot_row][index];
                }
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == width {
            break;
        }
    }
    if pivot_row != width {
        return Err("invertible matrix has dependent first-half row subset".into());
    }

    let pivot_words = width.div_ceil(64);
    let mut restriction = vec![vec![0u64; pivot_words]; width];
    for source_row in 0..width {
        let row = state
            .row_words(source_row as u32)
            .ok_or_else(|| "partial-shear source row is absent".to_owned())?;
        for (pivot_index, &column) in pivot_columns.iter().enumerate() {
            if row[column / 64] & (1u64 << (column % 64)) != 0 {
                restriction[source_row][pivot_index / 64] |= 1u64 << (pivot_index % 64);
            }
        }
    }
    let inverse = invert_packed_square(&restriction, width)?;

    let mut block: Vec<Vec<u8>> = Vec::with_capacity(targets.len());
    for &target in &targets {
        let row = state
            .row_words(target as u32)
            .ok_or_else(|| "partial-shear target row is absent".to_owned())?;
        let mut coefficients = vec![0u64; pivot_words];
        for (pivot_index, &column) in pivot_columns.iter().enumerate() {
            if row[column / 64] & (1u64 << (column % 64)) != 0 {
                for (word, value) in coefficients.iter_mut().enumerate() {
                    *value ^= inverse[pivot_index][word];
                }
            }
        }
        block.push(
            (0..width)
                .map(|source| ((coefficients[source / 64] >> (source % 64)) & 1) as u8)
                .collect(),
        );
    }

    if !block.iter().flatten().any(|bit| *bit != 0) {
        return Ok(None);
    }
    Ok(Some(ShearDecomposition {
        source_wires: sources,
        target_wires: targets,
        block,
        pivot_columns: Some(pivot_columns),
    }))
}

fn invert_packed_square(matrix: &[Vec<u64>], width: usize) -> Result<Vec<Vec<u64>>, String> {
    let words = width.div_ceil(64);
    if matrix.len() != width || matrix.iter().any(|row| row.len() != words) {
        return Err("partial-shear restriction matrix has invalid shape".into());
    }
    let mut left = matrix.to_vec();
    let mut right = vec![vec![0u64; words]; width];
    for row in 0..width {
        right[row][row / 64] |= 1u64 << (row % 64);
    }

    for column in 0..width {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (column..width).find(|&row| left[row][word] & bit != 0) else {
            return Err("partial-shear pivot restriction is singular".into());
        };
        left.swap(column, pivot);
        right.swap(column, pivot);
        for row in 0..width {
            if row != column && left[row][word] & bit != 0 {
                for index in 0..words {
                    left[row][index] ^= left[column][index];
                    right[row][index] ^= right[column][index];
                }
            }
        }
    }
    Ok(right)
}
