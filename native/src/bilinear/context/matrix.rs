use super::super::BilinearContextError;
use std::ops::Range;

pub(in crate::bilinear) fn insert_row_basis(pivots: &mut [Option<Vec<u64>>], row: &[u64]) -> bool {
    let mut reduced = row.to_vec();
    for column in (0..pivots.len()).rev() {
        if !get_bit(&reduced, column) {
            continue;
        }
        if let Some(pivot) = &pivots[column] {
            xor_words(&mut reduced, pivot);
        } else {
            pivots[column] = Some(reduced);
            return true;
        }
    }
    false
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::bilinear) struct BitMatrix {
    pub(in crate::bilinear) rows: Vec<Vec<u64>>,
    pub(in crate::bilinear) ncols: usize,
}

impl BitMatrix {
    pub(in crate::bilinear) fn from_packed(
        rows: Vec<Vec<u64>>,
        ncols: usize,
        label: &str,
    ) -> Result<Self, BilinearContextError> {
        let expected = words_for(ncols).ok_or_else(|| {
            BilinearContextError(format!("{label} width overflows packed storage"))
        })?;
        for row in &rows {
            if row.len() != expected {
                return Err(BilinearContextError(format!(
                    "{label} row uses {} words, expected {expected}",
                    row.len()
                )));
            }
            if ncols % 64 != 0
                && row
                    .last()
                    .is_some_and(|word| word & !((1u64 << (ncols % 64)) - 1) != 0)
            {
                return Err(BilinearContextError(format!(
                    "{label} row has nonzero padding bits"
                )));
            }
        }
        Ok(Self { rows, ncols })
    }

    pub(in crate::bilinear) fn from_rows(
        rows: Vec<Vec<u64>>,
        ncols: usize,
    ) -> Result<Self, String> {
        let expected = words_for(ncols).ok_or_else(|| "i05_matrix_width_overflow".to_string())?;
        if rows.iter().any(|row| row.len() != expected) {
            return Err("i05_internal_matrix_storage_shape_mismatch".into());
        }
        Ok(Self { rows, ncols })
    }

    pub(in crate::bilinear) fn zero(nrows: usize, ncols: usize) -> Result<Self, String> {
        let words = words_for(ncols).ok_or_else(|| "i05_matrix_width_overflow".to_string())?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(nrows)
            .map_err(|_| "i05_internal_matrix_allocation_failed".to_string())?;
        for _ in 0..nrows {
            let mut row = Vec::new();
            row.try_reserve_exact(words)
                .map_err(|_| "i05_internal_matrix_allocation_failed".to_string())?;
            row.resize(words, 0);
            rows.push(row);
        }
        Ok(Self { rows, ncols })
    }

    pub(in crate::bilinear) fn nrows(&self) -> usize {
        self.rows.len()
    }

    pub(in crate::bilinear) fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols)
    }

    pub(in crate::bilinear) fn rank(&self) -> usize {
        let mut pivots = vec![None; self.ncols];
        self.rows
            .iter()
            .filter(|row| insert_row_basis(&mut pivots, row))
            .count()
    }

    pub(in crate::bilinear) fn transpose(&self) -> Result<Self, String> {
        let mut transposed = Self::zero(self.ncols, self.nrows())?;
        for row in 0..self.nrows() {
            for column in 0..self.ncols {
                if get_bit(&self.rows[row], column) {
                    set_bit(&mut transposed.rows[column], row);
                }
            }
        }
        Ok(transposed)
    }

    pub(in crate::bilinear) fn multiply(&self, other: &Self) -> Result<Self, String> {
        if self.ncols != other.nrows() {
            return Err("i05_matrix_product_dimension_mismatch".into());
        }
        let mut product = Self::zero(self.nrows(), other.ncols)?;
        for (out, left) in product.rows.iter_mut().zip(&self.rows) {
            for inner in 0..self.ncols {
                if get_bit(left, inner) {
                    xor_words(out, &other.rows[inner]);
                }
            }
        }
        Ok(product)
    }

    pub(in crate::bilinear) fn xor(&self, other: &Self) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err("i05_matrix_sum_dimension_mismatch".into());
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(left, right)| left.iter().zip(right).map(|(a, b)| a ^ b).collect())
            .collect();
        Self::from_rows(rows, self.ncols)
    }

    pub(in crate::bilinear) fn select_rows(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.nrows() {
            return Err("i05_matrix_row_slice_out_of_range".into());
        }
        Self::from_rows(self.rows[range].to_vec(), self.ncols)
    }

    pub(in crate::bilinear) fn select_columns(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.ncols {
            return Err("i05_matrix_column_slice_out_of_range".into());
        }
        self.transpose()?.select_rows(range)?.transpose()
    }

    pub(in crate::bilinear) fn nullspace(&self) -> Result<Vec<Vec<u64>>, String> {
        let mut rows = self.rows.clone();
        let mut pivot_columns = Vec::new();
        let mut pivot_row = 0usize;
        for column in 0..self.ncols {
            let Some(found) = (pivot_row..rows.len()).find(|row| get_bit(&rows[*row], column))
            else {
                continue;
            };
            rows.swap(pivot_row, found);
            for row in 0..rows.len() {
                if row != pivot_row && get_bit(&rows[row], column) {
                    let pivot = rows[pivot_row].clone();
                    xor_words(&mut rows[row], &pivot);
                }
            }
            pivot_columns.push(column);
            pivot_row += 1;
        }

        let mut basis = Vec::new();
        for free_column in 0..self.ncols {
            if pivot_columns.contains(&free_column) {
                continue;
            }
            let mut vector = vec![
                0u64;
                words_for(self.ncols)
                    .ok_or_else(|| "i05_nullspace_width_overflow".to_string())?
            ];
            set_bit(&mut vector, free_column);
            for (row, pivot_column) in pivot_columns.iter().copied().enumerate() {
                if get_bit(&rows[row], free_column) {
                    set_bit(&mut vector, pivot_column);
                }
            }
            basis.push(vector);
        }
        Ok(basis)
    }

    pub(in crate::bilinear) fn inverse(&self) -> Option<Self> {
        if self.nrows() != self.ncols {
            return None;
        }
        let width = self.ncols.checked_mul(2)?;
        let mut augmented = Self::zero(self.nrows(), width).ok()?;
        for row in 0..self.nrows() {
            for column in 0..self.ncols {
                if get_bit(&self.rows[row], column) {
                    set_bit(&mut augmented.rows[row], column);
                }
            }
            set_bit(&mut augmented.rows[row], self.ncols + row);
        }

        for column in 0..self.ncols {
            let found =
                (column..self.nrows()).find(|row| get_bit(&augmented.rows[*row], column))?;
            augmented.rows.swap(column, found);
            for row in 0..self.nrows() {
                if row != column && get_bit(&augmented.rows[row], column) {
                    let pivot = augmented.rows[column].clone();
                    xor_words(&mut augmented.rows[row], &pivot);
                }
            }
        }

        let mut inverse = Self::zero(self.nrows(), self.ncols).ok()?;
        for row in 0..self.nrows() {
            for column in 0..self.ncols {
                if get_bit(&augmented.rows[row], self.ncols + column) {
                    set_bit(&mut inverse.rows[row], column);
                }
            }
        }
        Some(inverse)
    }

    pub(in crate::bilinear) fn from_columns(
        columns: Vec<Vec<u64>>,
        nrows: usize,
    ) -> Result<Self, String> {
        let ncols = columns.len();
        let expected = words_for(nrows).ok_or_else(|| "i05_column_width_overflow".to_string())?;
        if columns.iter().any(|column| {
            column.len() != expected
                || (nrows % 64 != 0
                    && column
                        .last()
                        .is_some_and(|word| word & !((1u64 << (nrows % 64)) - 1) != 0))
        }) {
            return Err("i05_column_vector_shape_mismatch".into());
        }
        let mut matrix = Self::zero(nrows, ncols)?;
        for (column_index, column) in columns.iter().enumerate() {
            for row_index in 0..nrows {
                if get_bit(column, row_index) {
                    set_bit(&mut matrix.rows[row_index], column_index);
                }
            }
        }
        Ok(matrix)
    }
}

pub(in crate::bilinear) fn words_for(bits: usize) -> Option<usize> {
    bits.checked_add(63).map(|value| value / 64)
}

fn get_bit(words: &[u64], index: usize) -> bool {
    words
        .get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

fn set_bit(words: &mut [u64], index: usize) {
    if let Some(word) = words.get_mut(index / 64) {
        *word |= 1u64 << (index % 64);
    }
}

fn xor_words(target: &mut [u64], source: &[u64]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target ^= source;
    }
}
