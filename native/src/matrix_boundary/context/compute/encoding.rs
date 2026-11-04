use crate::features::{FeatureComputation, FeatureValue};
use crate::matrix_boundary::{MatrixBoundarySpec, MATRIX_BOUNDARY_IDS, PORT_WIRE_ROLES};
use crate::packet::FeatureStatus;
use crate::schema::FeatureId;
use std::collections::HashMap;

pub(super) fn encode(
    spec: &MatrixBoundarySpec,
    n: usize,
    ports: usize,
    relation_rows: &[Vec<u64>],
    visible_rank: usize,
    relation_rank: usize,
    protected_rank: usize,
) -> HashMap<FeatureId, FeatureComputation> {
    let mut port_values = vec![0.0f32; ports * 2];
    let mut port_incidence = vec![0.0f32; ports * n * PORT_WIRE_ROLES.len()];
    for (index, wire) in spec.inputs.iter().enumerate() {
        port_values[index * 2] = 0.0;
        port_values[index * 2 + 1] = if spec.protected_inputs.binary_search(wire).is_ok() {
            1.0
        } else {
            0.0
        };
        set_incidence(&mut port_incidence, ports, n, index, *wire, 0);
        if spec.protected_inputs.binary_search(wire).is_ok() {
            set_incidence(&mut port_incidence, ports, n, index, *wire, 2);
        }
    }
    for (output_index, wire) in spec.outputs.iter().enumerate() {
        let group = spec.inputs.len() + output_index;
        port_values[group * 2] = 1.0;
        port_values[group * 2 + 1] = 0.0;
        set_incidence(&mut port_incidence, ports, n, group, *wire, 1);
    }

    let mut equation_values = Vec::with_capacity(relation_rank * 3);
    let mut equation_incidence = vec![0.0f32; relation_rank * n * PORT_WIRE_ROLES.len()];
    let slack = ports - relation_rank;
    for (equation_index, mask) in relation_rows.iter().enumerate() {
        equation_values.extend([relation_rank as f32, slack as f32, 0.0]);
        for (input_index, wire) in spec.inputs.iter().enumerate() {
            if crate::matrix_boundary::linear::bit_is_set(mask, input_index) {
                set_incidence(
                    &mut equation_incidence,
                    relation_rank,
                    n,
                    equation_index,
                    *wire,
                    0,
                );
                if spec.protected_inputs.binary_search(wire).is_ok() {
                    set_incidence(
                        &mut equation_incidence,
                        relation_rank,
                        n,
                        equation_index,
                        *wire,
                        2,
                    );
                }
            }
        }
        for (output_index, wire) in spec.outputs.iter().enumerate() {
            if crate::matrix_boundary::linear::bit_is_set(mask, spec.inputs.len() + output_index) {
                set_incidence(
                    &mut equation_incidence,
                    relation_rank,
                    n,
                    equation_index,
                    *wire,
                    1,
                );
            }
        }
    }

    let mut values = HashMap::with_capacity(MATRIX_BOUNDARY_IDS.len());
    insert_computed(
        &mut values,
        (b'C', 8),
        vec![1],
        vec![(spec.inputs.len() - visible_rank) as f32],
        vec![],
        vec![],
    );
    insert_computed(
        &mut values,
        (b'I', 1),
        vec![ports, 2],
        port_values,
        vec![1; ports],
        port_incidence,
    );
    insert_computed(
        &mut values,
        (b'I', 2),
        vec![relation_rank, 3],
        equation_values,
        vec![1; relation_rank],
        equation_incidence,
    );
    insert_computed(
        &mut values,
        (b'I', 3),
        vec![2],
        vec![
            if protected_rank > 0 { 1.0 } else { 0.0 },
            protected_rank as f32,
        ],
        vec![],
        vec![],
    );
    let i04 = if spec.weights.is_some() && spec.zero_outputs.is_empty() {
        FeatureComputation::Computed(FeatureValue {
            shape: vec![3],
            values: vec![
                visible_rank as f32,
                visible_rank as f32,
                (spec.inputs.len() - visible_rank) as f32,
            ],
            token_valid: vec![],
            incidence: vec![],
        })
    } else {
        FeatureComputation::Unavailable {
            status: FeatureStatus::Unsupported,
            reason: "i04_requires_positive_product_weights_and_no_pinned_zero_outputs".into(),
        }
    };
    values.insert(
        FeatureId::new(b'I', 4).expect("I04 is in the canonical manifest"),
        i04,
    );
    insert_computed(
        &mut values,
        (b'I', 11),
        vec![4],
        vec![ports as f32, (ports - relation_rank) as f32, 1.0, 0.0],
        vec![],
        vec![],
    );
    values
}

fn insert_computed(
    values: &mut HashMap<FeatureId, FeatureComputation>,
    key: (u8, u8),
    shape: Vec<usize>,
    values_out: Vec<f32>,
    token_valid: Vec<u8>,
    incidence: Vec<f32>,
) {
    let id = FeatureId::new(key.0, key.1).expect("matrix-boundary ID is canonical");
    values.insert(
        id,
        FeatureComputation::Computed(FeatureValue {
            shape,
            values: values_out,
            token_valid,
            incidence,
        }),
    );
}

fn set_incidence(
    incidence: &mut [f32],
    groups: usize,
    n: usize,
    group: usize,
    wire: u32,
    role: usize,
) {
    let index = (group * n + wire as usize) * PORT_WIRE_ROLES.len() + role;
    debug_assert!(group < groups && index < incidence.len());
    incidence[index] = 1.0;
}
