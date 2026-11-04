use crate::matrix_boundary::linear::{
    canonicalize_rows, matrix_bit, protected_projection_rank, rank_binary_rows, set_bit,
    words_for_bits,
};
use crate::matrix_boundary::MatrixBoundarySpec;
use crate::state::MatrixState;

pub(super) struct RelationSummary {
    pub(super) relation_rows: Vec<Vec<u64>>,
    pub(super) visible_rank: usize,
    pub(super) relation_rank: usize,
    pub(super) protected_rank: usize,
}

pub(super) fn build(
    spec: &MatrixBoundarySpec,
    state: &MatrixState,
    ports: usize,
    equations: usize,
) -> Result<RelationSummary, &'static str> {
    let mut mapped_rows = Vec::with_capacity(spec.outputs.len());
    for output in &spec.outputs {
        let Some(row) = state.row_words(*output) else {
            return Err("selected_output_row_is_outside_matrix");
        };
        let mut projected = vec![0u64; words_for_bits(spec.inputs.len())];
        for (input_index, wire) in spec.inputs.iter().enumerate() {
            if matrix_bit(row, *wire as usize) {
                set_bit(&mut projected, input_index);
            }
        }
        mapped_rows.push(projected);
    }
    let visible_rank = rank_binary_rows(mapped_rows.clone(), spec.inputs.len());

    let mut relation_rows = Vec::with_capacity(equations);
    for (output_index, row) in mapped_rows.iter().enumerate() {
        let mut equation = vec![0u64; words_for_bits(ports)];
        for bit in 0..spec.inputs.len() {
            if crate::matrix_boundary::linear::bit_is_set(row, bit) {
                set_bit(&mut equation, bit);
            }
        }
        set_bit(&mut equation, spec.inputs.len() + output_index);
        relation_rows.push(equation);
    }
    for output in &spec.zero_outputs {
        let output_index = spec
            .outputs
            .binary_search(output)
            .expect("validated output");
        let mut equation = vec![0u64; words_for_bits(ports)];
        set_bit(&mut equation, spec.inputs.len() + output_index);
        relation_rows.push(equation);
    }
    let relation_rows = canonicalize_rows(relation_rows, ports);
    let relation_rank = relation_rows.len();
    let protected_rank = protected_projection_rank(spec, &relation_rows, ports);
    Ok(RelationSummary {
        relation_rows,
        visible_rank,
        relation_rank,
        protected_rank,
    })
}
