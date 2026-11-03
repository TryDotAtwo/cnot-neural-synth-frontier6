use super::linear::{
    bit_is_set, canonicalize_rows, matrix_bit, protected_projection_rank, rank_binary_rows,
    set_bit, words_for_bits,
};
use super::{CACHE_CAPACITY, MATRIX_BOUNDARY_IDS, MatrixBoundarySpec, PORT_WIRE_ROLES};
use crate::features::{FeatureComputation, FeatureValue};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::FeatureId;
use crate::state::MatrixState;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

#[derive(Clone)]
struct BoundaryCacheEntry {
    state: StateIdentity,
    max_bytes: Option<usize>,
    values: HashMap<FeatureId, FeatureComputation>,
}

pub(super) struct MatrixBoundaryContext {
    specs: Vec<MatrixBoundarySpec>,
    max_ports: usize,
    max_equations: usize,
    cache: Mutex<VecDeque<BoundaryCacheEntry>>,
}

impl MatrixBoundaryContext {
    pub(super) fn new(
        specs: Vec<MatrixBoundarySpec>,
        max_ports: usize,
        max_equations: usize,
    ) -> Self {
        Self {
            specs,
            max_ports,
            max_equations,
            cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
        }
    }

    pub(super) fn values_for(
        &self,
        state: &MatrixState,
        max_bytes: Option<usize>,
    ) -> HashMap<FeatureId, FeatureComputation> {
        let identity = StateIdentity::from(state);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache
                .iter()
                .position(|entry| entry.state == identity && entry.max_bytes == max_bytes)
            {
                if let Some(entry) = cache.remove(index) {
                    let result = entry.values.clone();
                    cache.push_front(entry);
                    return result;
                }
            }
        }

        let values = self.compute(state, max_bytes);
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(BoundaryCacheEntry {
                state: identity,
                max_bytes,
                values: values.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        values
    }

    fn compute(
        &self,
        state: &MatrixState,
        max_bytes: Option<usize>,
    ) -> HashMap<FeatureId, FeatureComputation> {
        let Some(spec) = self.specs.iter().find(|spec| spec.n == state.dimension()) else {
            return unavailable_all(
                FeatureStatus::Unsupported,
                "no_explicit_boundary_for_matrix_width",
            );
        };
        let ports = spec.inputs.len().saturating_add(spec.outputs.len());
        let equations = spec.outputs.len().saturating_add(spec.zero_outputs.len());
        if ports > self.max_ports || equations > self.max_equations {
            return unavailable_all(
                FeatureStatus::ResourceLimit,
                "matrix_boundary_port_or_equation_budget_exceeded",
            );
        }

        let mut mapped_rows = Vec::with_capacity(spec.outputs.len());
        for output in &spec.outputs {
            let Some(row) = state.row_words(*output) else {
                return unavailable_all(
                    FeatureStatus::Unsupported,
                    "selected_output_row_is_outside_matrix",
                );
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
                if bit_is_set(row, bit) {
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

        let Some(byte_count) =
            estimated_output_bytes(ports, state.dimension() as usize, relation_rank)
        else {
            return unavailable_all(
                FeatureStatus::ResourceLimit,
                "matrix_boundary_output_size_overflow",
            );
        };
        if max_bytes.is_some_and(|budget| byte_count > budget) {
            return unavailable_all(
                FeatureStatus::ResourceLimit,
                "matrix_boundary_output_exceeds_policy_byte_budget",
            );
        }

        let n = state.dimension() as usize;
        let mut port_values = vec![0.0f32; ports * 2];
        let mut port_incidence = vec![0.0f32; ports * n * PORT_WIRE_ROLES.len()];
        for (index, wire) in spec.inputs.iter().enumerate() {
            port_values[index * 2] = 0.0; // input port kind
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
            port_values[group * 2] = 1.0; // output port kind
            port_values[group * 2 + 1] = 0.0;
            set_incidence(&mut port_incidence, ports, n, group, *wire, 1);
        }

        let mut equation_values = Vec::with_capacity(relation_rank * 3);
        let mut equation_incidence = vec![0.0f32; relation_rank * n * PORT_WIRE_ROLES.len()];
        let slack = ports - relation_rank;
        for (equation_index, mask) in relation_rows.iter().enumerate() {
            equation_values.extend([relation_rank as f32, slack as f32, 0.0]);
            for (input_index, wire) in spec.inputs.iter().enumerate() {
                if bit_is_set(mask, input_index) {
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
                if bit_is_set(mask, spec.inputs.len() + output_index) {
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

fn unavailable_all(status: FeatureStatus, reason: &str) -> HashMap<FeatureId, FeatureComputation> {
    MATRIX_BOUNDARY_IDS
        .iter()
        .map(|(family, ordinal)| {
            (
                FeatureId::new(*family, *ordinal).expect("matrix-boundary ID is canonical"),
                FeatureComputation::Unavailable {
                    status,
                    reason: reason.into(),
                },
            )
        })
        .collect()
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

fn estimated_output_bytes(ports: usize, n: usize, equations: usize) -> Option<usize> {
    let group_incidence = ports.checked_mul(n)?.checked_mul(PORT_WIRE_ROLES.len())?;
    let equation_incidence = equations
        .checked_mul(n)?
        .checked_mul(PORT_WIRE_ROLES.len())?;
    let float_count = ports
        .checked_mul(2)?
        .checked_add(group_incidence)?
        .checked_add(equations.checked_mul(3)?)?
        .checked_add(equation_incidence)?
        .checked_add(1 + 2 + 3 + 4)?;
    float_count
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_add(ports)?
        .checked_add(equations)
}
