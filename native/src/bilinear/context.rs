use super::producer::reduce_stage55;
use super::BilinearContextError;
use crate::state::MatrixState;
use std::collections::HashSet;

mod matrix;
pub(super) use matrix::{insert_row_basis, words_for, BitMatrix};

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
