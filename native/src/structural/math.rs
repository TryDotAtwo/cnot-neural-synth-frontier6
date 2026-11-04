//! Shared allocation, matrix, and graph kernels for structural features.

mod bit_matrix;
mod dense;
mod graph;

pub(crate) use bit_matrix::gf2_inverse;
pub(in crate::structural) use bit_matrix::{gf2_rank, packed_to_dense};
pub(in crate::structural) use dense::{col_sums, hadamard, matmul, row_sums, transpose_dense};
pub(in crate::structural) use graph::{
    dag_depth, equivalence_classes, transitive_closure, undirected_adjacency, undirected_components,
};

pub(super) fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("feature allocation failed: {error}"))?;
    values.resize(len, T::default());
    Ok(values)
}
