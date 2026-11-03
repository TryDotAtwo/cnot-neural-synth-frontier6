use super::BilinearContextError;
use super::producer::reduce_stage55;
use crate::state::MatrixState;
use std::collections::HashSet;
use std::ops::Range;

/// Explicit physical row/column selection and the four labelled port matrices.
///
/// Matrix rows are packed into little-endian `u64` words: column zero is bit
/// zero of the first word. Shapes are `U[p,t]`, `V[t,q]`, `A0[p,r]`, and
/// `D0[m,q]`; `rows` and `columns` select the `m x r` physical block `C`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BilinearContext {
    pub(super) n: u32,
    pub(super) rows: Vec<u32>,
    pub(super) columns: Vec<u32>,
    pub(super) hidden_width: usize,
    pub(super) q: usize,
    pub(super) u: BitMatrix,
    pub(super) v: BitMatrix,
    pub(super) a0: BitMatrix,
    pub(super) d0: BitMatrix,
}

impl BilinearContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: u32,
        rows: Vec<u32>,
        columns: Vec<u32>,
        hidden_width: usize,
        u_rows: Vec<Vec<u64>>,
        v_rows: Vec<Vec<u64>>,
        q: usize,
        a0_rows: Vec<Vec<u64>>,
        d0_rows: Vec<Vec<u64>>,
    ) -> Result<Self, BilinearContextError> {
        if n == 0 {
            return Err(BilinearContextError(
                "positive physical matrix width required".into(),
            ));
        }
        validate_physical_indices(&rows, n, "selected rows")?;
        validate_physical_indices(&columns, n, "selected columns")?;

        let p = u_rows.len();
        let m = rows.len();
        let r = columns.len();
        let u = BitMatrix::from_packed(u_rows, hidden_width, "U")?;
        let v = BitMatrix::from_packed(v_rows, q, "V")?;
        let a0 = BitMatrix::from_packed(a0_rows, r, "A0")?;
        let d0 = BitMatrix::from_packed(d0_rows, q, "D0")?;
        let t = hidden_width;

        if u.nrows() != p || v.nrows() != t || a0.nrows() != p || d0.nrows() != m {
            return Err(BilinearContextError(
                "port matrix row counts do not match the declared dimensions".into(),
            ));
        }
        if u.rank() != p {
            return Err(BilinearContextError("U must have independent rows".into()));
        }
        if v.rank() != q {
            return Err(BilinearContextError(
                "V must have independent columns".into(),
            ));
        }

        Ok(Self {
            n,
            rows,
            columns,
            hidden_width,
            q,
            u,
            v,
            a0,
            d0,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub(super) fn dimensions(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.rows.len(),
            self.columns.len(),
            self.u.nrows(),
            self.hidden_width,
            self.q,
        )
    }

    pub(super) fn matrix_bit_budget(&self) -> Option<usize> {
        let (m, r, p, t, q) = self.dimensions();
        p.checked_mul(t)?
            .checked_add(t.checked_mul(q)?)?
            .checked_add(p.checked_mul(r)?)?
            .checked_add(m.checked_mul(q)?)?
            .checked_add(m.checked_mul(r)?)
    }

    pub(super) fn working_bytes_upper_bound(&self) -> Option<usize> {
        let (m, r, p, t, q) = self.dimensions();
        let n = usize::try_from(self.n).ok()?;
        let side = n.max(m).max(r).max(p).max(t).max(q).max(1);
        let square = side.checked_mul(side)?;
        n.checked_mul(n)?
            .checked_add(m.checked_mul(r)?)?
            .checked_add(square)?
            .checked_add(1)?
            .checked_mul(128)
    }

    pub(super) fn source_values(&self, state: &MatrixState) -> Result<[usize; 2], String> {
        if state.dimension() != self.n {
            return Err("physical_width_does_not_match_bilinear_context".into());
        }
        let m = self.rows.len();
        let r = self.columns.len();
        let mut c_rows = Vec::new();
        c_rows
            .try_reserve_exact(m)
            .map_err(|_| "i05_physical_block_allocation_failed".to_string())?;
        let words = words_for(r).ok_or_else(|| "i05_block_width_overflow".to_string())?;
        for physical_row in &self.rows {
            let row = state
                .row_words(*physical_row)
                .ok_or_else(|| "i05_selected_physical_row_is_out_of_range".to_string())?;
            let mut packed = vec![0u64; words];
            for (selected_column, physical_column) in self.columns.iter().copied().enumerate() {
                let word = usize::try_from(physical_column / 64)
                    .map_err(|_| "i05_selected_column_index_overflow".to_string())?;
                let bit = physical_column % 64;
                if row
                    .get(word)
                    .is_some_and(|value| value & (1u64 << bit) != 0)
                {
                    packed[selected_column / 64] |= 1u64 << (selected_column % 64);
                }
            }
            c_rows.push(packed);
        }
        let c = BitMatrix::from_packed(c_rows, r, "C").map_err(|error| error.0)?;
        let reduction = reduce_stage55(self, &c)?;
        Ok([reduction.residual_rank, reduction.hidden])
    }
}

fn validate_physical_indices(
    indices: &[u32],
    n: u32,
    label: &str,
) -> Result<(), BilinearContextError> {
    if indices.is_empty() {
        return Err(BilinearContextError(format!("{label} must be nonempty")));
    }
    let mut seen = HashSet::with_capacity(indices.len());
    if indices
        .iter()
        .any(|index| *index >= n || !seen.insert(*index))
    {
        return Err(BilinearContextError(format!(
            "{label} must contain distinct in-range physical indices"
        )));
    }
    Ok(())
}

pub(super) fn insert_row_basis(pivots: &mut [Option<Vec<u64>>], row: &[u64]) -> bool {
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
pub(super) struct BitMatrix {
    pub(super) rows: Vec<Vec<u64>>,
    pub(super) ncols: usize,
}

impl BitMatrix {
    pub(super) fn from_packed(
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

    pub(super) fn from_rows(rows: Vec<Vec<u64>>, ncols: usize) -> Result<Self, String> {
        let expected = words_for(ncols).ok_or_else(|| "i05_matrix_width_overflow".to_string())?;
        if rows.iter().any(|row| row.len() != expected) {
            return Err("i05_internal_matrix_storage_shape_mismatch".into());
        }
        Ok(Self { rows, ncols })
    }

    pub(super) fn zero(nrows: usize, ncols: usize) -> Result<Self, String> {
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

    pub(super) fn nrows(&self) -> usize {
        self.rows.len()
    }

    pub(super) fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols)
    }

    pub(super) fn rank(&self) -> usize {
        let mut pivots = vec![None; self.ncols];
        self.rows
            .iter()
            .filter(|row| insert_row_basis(&mut pivots, row))
            .count()
    }

    pub(super) fn transpose(&self) -> Result<Self, String> {
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

    pub(super) fn multiply(&self, other: &Self) -> Result<Self, String> {
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

    pub(super) fn xor(&self, other: &Self) -> Result<Self, String> {
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

    pub(super) fn select_rows(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.nrows() {
            return Err("i05_matrix_row_slice_out_of_range".into());
        }
        Self::from_rows(self.rows[range].to_vec(), self.ncols)
    }

    pub(super) fn select_columns(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.ncols {
            return Err("i05_matrix_column_slice_out_of_range".into());
        }
        self.transpose()?.select_rows(range)?.transpose()
    }

    pub(super) fn nullspace(&self) -> Result<Vec<Vec<u64>>, String> {
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

    pub(super) fn inverse(&self) -> Option<Self> {
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

    pub(super) fn from_columns(columns: Vec<Vec<u64>>, nrows: usize) -> Result<Self, String> {
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

pub(super) fn words_for(bits: usize) -> Option<usize> {
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
