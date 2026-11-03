use super::context::{BilinearContext, BitMatrix, insert_row_basis, words_for};
use super::{DEFAULT_MAX_DIMENSION, DEFAULT_MAX_MATRIX_BITS, STAGE55_SOURCE_SHA256};
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, feature_spec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

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

pub(super) struct Stage55Reduction {
    pub(super) residual_rank: usize,
    pub(super) hidden: usize,
}

pub(super) fn reduce_stage55(
    context: &BilinearContext,
    c: &BitMatrix,
) -> Result<Stage55Reduction, String> {
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
