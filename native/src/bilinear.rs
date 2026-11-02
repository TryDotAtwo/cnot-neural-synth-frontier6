//! Exact source-defined I05 port reduction for a labelled binary matrix block.
//!
//! The two emitted channels are the rank of the stage-55 residual and the
//! dimension of its free hidden space. They describe a finite matrix-product
//! fiber; they are not CNOT distance or a count of physical circuits.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::ops::Range;

/// SHA-256 of `research_v3/artifacts/frontier61_source_v1/stage55_bilinear_kernel/bilinear_kernel.py`.
pub const STAGE55_SOURCE_SHA256: &str =
    "30c90c396e8f5f1d43921f09a0d07ca723afa6e38a9c0ebed4f68b5b2aab5353";

const DEFAULT_MAX_DIMENSION: usize = 144;
const DEFAULT_MAX_MATRIX_BITS: usize = 100_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BilinearContextError(pub String);

impl fmt::Display for BilinearContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BilinearContextError {}

/// Explicit physical row/column selection and the four labelled port matrices.
///
/// Matrix rows are packed into little-endian `u64` words: column zero is bit
/// zero of the first word. Shapes are `U[p,t]`, `V[t,q]`, `A0[p,r]`, and
/// `D0[m,q]`; `rows` and `columns` select the `m x r` physical block `C`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BilinearContext {
    n: u32,
    rows: Vec<u32>,
    columns: Vec<u32>,
    hidden_width: usize,
    q: usize,
    u: BitMatrix,
    v: BitMatrix,
    a0: BitMatrix,
    d0: BitMatrix,
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

    fn dimensions(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.rows.len(),
            self.columns.len(),
            self.u.nrows(),
            self.hidden_width,
            self.q,
        )
    }

    fn matrix_bit_budget(&self) -> Option<usize> {
        let (m, r, p, t, q) = self.dimensions();
        p.checked_mul(t)?
            .checked_add(t.checked_mul(q)?)?
            .checked_add(p.checked_mul(r)?)?
            .checked_add(m.checked_mul(q)?)?
            .checked_add(m.checked_mul(r)?)
    }

    fn working_bytes_upper_bound(&self) -> Option<usize> {
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

    fn source_values(&self, state: &MatrixState) -> Result<[usize; 2], String> {
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

struct BilinearProducer {
    spec: FeatureSpec,
    contexts: Vec<BilinearContext>,
    max_dimension: usize,
    max_matrix_bits: usize,
}

impl FeatureProducer for BilinearProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "i05_requires_augmentation_only_symmetry",
            );
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension())
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i05_context_for_matrix_width",
            );
        };

        let (m, r, p, t, q) = context.dimensions();
        let largest = [m, r, p, t, q].into_iter().max().unwrap_or_default();
        if largest > self.max_dimension
            || context
                .matrix_bit_budget()
                .is_none_or(|bits| bits > self.max_matrix_bits)
        {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i05_dimension_or_matrix_bit_budget_exceeded",
            );
        }
        if policy.max_bytes.is_some_and(|limit| {
            context
                .working_bytes_upper_bound()
                .is_none_or(|estimate| estimate > limit)
        }) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "i05_working_memory_budget_exceeded_before_matrix_allocation",
            );
        }

        match context.source_values(state) {
            Ok(values) if values.iter().all(|value| *value <= (1 << 24)) => {
                FeatureComputation::Computed(FeatureValue {
                    shape: vec![2],
                    values: values.into_iter().map(|value| value as f32).collect(),
                    token_valid: Vec::new(),
                    incidence: Vec::new(),
                })
            }
            Ok(_) => unavailable(
                FeatureStatus::ResourceLimit,
                "i05_integer_value_exceeds_exact_f32_range",
            ),
            Err(reason) => unavailable(FeatureStatus::ResourceLimit, &reason),
        }
    }
}

struct Stage55Reduction {
    residual_rank: usize,
    hidden: usize,
}

fn reduce_stage55(context: &BilinearContext, c: &BitMatrix) -> Result<Stage55Reduction, String> {
    let (m, r, p, t, q) = context.dimensions();
    if c.shape() != (m, r) {
        return Err("i05_selected_block_shape_mismatch".into());
    }

    let uv = context.u.multiply(&context.v)?;
    let uv_columns = uv.transpose()?.rows;
    let v_columns = context.v.transpose()?.rows;
    let mut image_basis = vec![None; p];
    let mut selected = Vec::new();
    let mut e1 = Vec::new();
    let mut coeff1 = Vec::new();
    for (column, image_vector) in uv_columns.iter().enumerate() {
        if insert_row_basis(&mut image_basis, image_vector) {
            selected.push(column);
            e1.push(v_columns[column].clone());
            coeff1.push(basis_vector(q, column)?);
        }
    }
    let h = selected.len();
    let coeff2 = uv.nullspace()?;

    let u_columns = context.u.transpose()?.rows;
    let mut e3 = Vec::new();
    for (coordinate, image_vector) in u_columns.iter().enumerate() {
        if insert_row_basis(&mut image_basis, image_vector) {
            e3.push(basis_vector(t, coordinate)?);
        }
    }
    if image_basis.iter().filter(|pivot| pivot.is_some()).count() != p {
        return Err("i05_source_port_basis_construction_failed".into());
    }

    let mut source_basis = e1;
    source_basis.extend(e3);
    let h_map = context
        .u
        .multiply(&BitMatrix::from_columns(source_basis, t)?)?;
    let h_map_inverse = h_map
        .inverse()
        .ok_or_else(|| "i05_source_port_basis_map_is_not_invertible".to_string())?;
    let a_split = h_map_inverse.multiply(&context.a0)?;
    let a1 = a_split.select_rows(0..h)?;
    let a3 = a_split.select_rows(h..p)?;

    let mut delivery_basis = coeff1;
    delivery_basis.extend(coeff2);
    let d_split = context
        .d0
        .multiply(&BitMatrix::from_columns(delivery_basis, q)?)?;
    let d1 = d_split.select_columns(0..h)?;
    let d2 = d_split.select_columns(h..q)?;
    let hidden = t
        .checked_add(h)
        .and_then(|value| value.checked_sub(p))
        .and_then(|value| value.checked_sub(q))
        .ok_or_else(|| "i05_source_reduction_has_negative_hidden_dimension".to_string())?;

    let left = BitMatrix::from_rows(d2.transpose()?.nullspace()?, m)?;
    let right = BitMatrix::from_columns(a3.nullspace()?, r)?;
    let delivered = d1.multiply(&a1)?;
    let residual_input = c.xor(&delivered)?;
    let residual = left.multiply(&residual_input)?.multiply(&right)?;

    // The I05 packet exposes only residual rank and free hidden width; the
    // alpha/beta ranks used by the source's count channel are not needed here.
    Ok(Stage55Reduction {
        residual_rank: residual.rank(),
        hidden,
    })
}

fn basis_vector(bits: usize, index: usize) -> Result<Vec<u64>, String> {
    if index >= bits {
        return Err("i05_basis_coordinate_out_of_range".into());
    }
    let mut vector = vec![0u64; words_for(bits).ok_or_else(|| "i05_width_overflow".to_string())?];
    vector[index / 64] |= 1u64 << (index % 64);
    Ok(vector)
}

fn insert_row_basis(pivots: &mut [Option<Vec<u64>>], row: &[u64]) -> bool {
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

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

/// Registers I05 for one or more explicit labelled matrix widths.
pub fn register_bilinear_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<BilinearContext>,
    max_dimension: usize,
    max_matrix_bits: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || max_dimension == 0 || max_matrix_bits == 0 {
        return Err(FeatureBuildError(
            "I05 requires explicit contexts and positive dimension/matrix-bit budgets".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I05 allows one labelled port context per physical width".into(),
        ));
    }

    let producer_version = producer_version(&contexts, max_dimension, max_matrix_bits);
    let id = FeatureId::new(b'I', 5).expect("I05 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 2;
    let spec = spec.with_provider_metadata(
        concat!(
            "explicit_stage55_UA=A0_DV=D0_DA=C_matrix_product_fiber;",
            "C=current_labelled_physical_block;residual_rank_and_free_hidden_width;",
            "augmentation_only;not_CNOT_distance"
        ),
        producer_version,
        ["M", "physical_block", "independent_U_V", "labelled_A0_D0"],
        "exact",
        "T2",
        std::iter::empty::<String>(),
        false,
    );
    registry.register(Box::new(BilinearProducer {
        spec,
        contexts,
        max_dimension,
        max_matrix_bits,
    }))
}

pub fn register_default_bilinear_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<BilinearContext>,
) -> Result<(), FeatureBuildError> {
    register_bilinear_contexts(
        registry,
        contexts,
        DEFAULT_MAX_DIMENSION,
        DEFAULT_MAX_MATRIX_BITS,
    )
}

fn producer_version(
    contexts: &[BilinearContext],
    max_dimension: usize,
    max_matrix_bits: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"stage55_i05_port_reduction_v1");
    hash_field(&mut digest, STAGE55_SOURCE_SHA256.as_bytes());
    hash_field(&mut digest, &max_dimension.to_le_bytes());
    hash_field(&mut digest, &max_matrix_bits.to_le_bytes());
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_field(&mut digest, &(context.hidden_width as u64).to_le_bytes());
        hash_field(&mut digest, &(context.q as u64).to_le_bytes());
        hash_indices(&mut digest, &context.rows);
        hash_indices(&mut digest, &context.columns);
        hash_matrix(&mut digest, &context.u);
        hash_matrix(&mut digest, &context.v);
        hash_matrix(&mut digest, &context.a0);
        hash_matrix(&mut digest, &context.d0);
    }
    format!(
        "stage55_i05_port_reduction_v1:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    hash_field(digest, &(indices.len() as u64).to_le_bytes());
    for index in indices {
        hash_field(digest, &index.to_le_bytes());
    }
}

fn hash_matrix(digest: &mut Sha256, matrix: &BitMatrix) {
    hash_field(digest, &(matrix.nrows() as u64).to_le_bytes());
    hash_field(digest, &(matrix.ncols as u64).to_le_bytes());
    for row in &matrix.rows {
        hash_field(digest, &(row.len() as u64).to_le_bytes());
        for word in row {
            hash_field(digest, &word.to_le_bytes());
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BitMatrix {
    rows: Vec<Vec<u64>>,
    ncols: usize,
}

impl BitMatrix {
    fn from_packed(
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

    fn from_rows(rows: Vec<Vec<u64>>, ncols: usize) -> Result<Self, String> {
        let expected = words_for(ncols).ok_or_else(|| "i05_matrix_width_overflow".to_string())?;
        if rows.iter().any(|row| row.len() != expected) {
            return Err("i05_internal_matrix_storage_shape_mismatch".into());
        }
        Ok(Self { rows, ncols })
    }

    fn zero(nrows: usize, ncols: usize) -> Result<Self, String> {
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

    fn nrows(&self) -> usize {
        self.rows.len()
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols)
    }

    fn rank(&self) -> usize {
        let mut pivots = vec![None; self.ncols];
        self.rows
            .iter()
            .filter(|row| insert_row_basis(&mut pivots, row))
            .count()
    }

    fn transpose(&self) -> Result<Self, String> {
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

    fn multiply(&self, other: &Self) -> Result<Self, String> {
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

    fn xor(&self, other: &Self) -> Result<Self, String> {
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

    fn select_rows(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.nrows() {
            return Err("i05_matrix_row_slice_out_of_range".into());
        }
        Self::from_rows(self.rows[range].to_vec(), self.ncols)
    }

    fn select_columns(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.ncols {
            return Err("i05_matrix_column_slice_out_of_range".into());
        }
        self.transpose()?.select_rows(range)?.transpose()
    }

    fn nullspace(&self) -> Result<Vec<Vec<u64>>, String> {
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

    fn inverse(&self) -> Option<Self> {
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

    fn from_columns(columns: Vec<Vec<u64>>, nrows: usize) -> Result<Self, String> {
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

fn words_for(bits: usize) -> Option<usize> {
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
