//! Validated construction and shape access for packed GF(2) matrices.

use super::super::super::BilinearContextError;
use super::packed::{get_bit, set_bit, words_for};
use super::BitMatrix;

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
            if !ncols.is_multiple_of(64)
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

    pub(in crate::bilinear) fn from_columns(
        columns: Vec<Vec<u64>>,
        nrows: usize,
    ) -> Result<Self, String> {
        let ncols = columns.len();
        let expected = words_for(nrows).ok_or_else(|| "i05_column_width_overflow".to_string())?;
        if columns.iter().any(|column| {
            column.len() != expected
                || (!nrows.is_multiple_of(64)
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
