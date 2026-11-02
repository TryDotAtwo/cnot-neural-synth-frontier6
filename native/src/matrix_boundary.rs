//! Exact source-defined features for an explicitly selected linear boundary.
//!
//! This is the Rust port of research_v3.features.matrix_boundary. It models
//! y = M[outputs, inputs] x with all other inputs fixed to zero. It does not
//! infer a boundary from a matrix and does not claim a generic circuit cost.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

const CACHE_CAPACITY: usize = 32;
const PORT_WIRE_ROLES: [&str; 3] = ["input", "output", "protected_initial"];
const MATRIX_BOUNDARY_IDS: [(u8, u8); 6] = [
    (b'C', 8),
    (b'I', 1),
    (b'I', 2),
    (b'I', 3),
    (b'I', 4),
    (b'I', 11),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBoundaryError(pub String);

impl fmt::Display for MatrixBoundaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MatrixBoundaryError {}

impl From<FeatureBuildError> for MatrixBoundaryError {
    fn from(error: FeatureBuildError) -> Self {
        Self(error.0)
    }
}

/// Nonnegative exact rational weight used only to validate/bind I04's source
/// input contract. The source feature emits a rank descriptor and never
/// approximates or numerically evaluates these weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalWeight {
    numerator: String,
    denominator: String,
}

impl RationalWeight {
    pub fn new(
        numerator: impl ToString,
        denominator: impl ToString,
    ) -> Result<Self, MatrixBoundaryError> {
        let numerator = canonical_unsigned_integer(&numerator.to_string())?;
        let denominator = canonical_unsigned_integer(&denominator.to_string())?;
        if denominator == "0" {
            return Err(MatrixBoundaryError(
                "rational weight denominator must be positive".into(),
            ));
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }
}

fn canonical_unsigned_integer(value: &str) -> Result<String, MatrixBoundaryError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(MatrixBoundaryError(
            "rational weights require nonnegative integer numerators and denominators".into(),
        ));
    }
    let digits = value.trim_start_matches('0');
    Ok(if digits.is_empty() {
        "0".into()
    } else {
        digits.into()
    })
}

/// A fixed physical boundary. All wire lists must be sorted, unique, and in
/// range. `weights`, when supplied, has one pair per output wire.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBoundarySpec {
    n: u32,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    protected_inputs: Vec<u32>,
    zero_outputs: Vec<u32>,
    weights: Option<Vec<(RationalWeight, RationalWeight)>>,
}

impl MatrixBoundarySpec {
    pub fn new(n: u32, inputs: Vec<u32>, outputs: Vec<u32>) -> Result<Self, MatrixBoundaryError> {
        let value = Self {
            n,
            inputs,
            outputs,
            protected_inputs: Vec::new(),
            zero_outputs: Vec::new(),
            weights: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_protected_inputs(
        mut self,
        protected_inputs: Vec<u32>,
    ) -> Result<Self, MatrixBoundaryError> {
        self.protected_inputs = protected_inputs;
        self.validate()?;
        Ok(self)
    }

    pub fn with_zero_outputs(
        mut self,
        zero_outputs: Vec<u32>,
    ) -> Result<Self, MatrixBoundaryError> {
        self.zero_outputs = zero_outputs;
        self.validate()?;
        Ok(self)
    }

    pub fn with_weights(
        mut self,
        weights: Vec<(RationalWeight, RationalWeight)>,
    ) -> Result<Self, MatrixBoundaryError> {
        if weights.len() != self.outputs.len() {
            return Err(MatrixBoundaryError(
                "one exact rational weight pair per output is required".into(),
            ));
        }
        self.weights = Some(weights);
        Ok(self)
    }

    fn validate(&self) -> Result<(), MatrixBoundaryError> {
        if self.n == 0 {
            return Err(MatrixBoundaryError(
                "positive physical width required".into(),
            ));
        }
        if self.inputs.is_empty() || self.outputs.is_empty() {
            return Err(MatrixBoundaryError(
                "nonempty input and output boundaries required".into(),
            ));
        }
        for (name, wires) in [
            ("inputs", &self.inputs),
            ("outputs", &self.outputs),
            ("protected inputs", &self.protected_inputs),
            ("zero outputs", &self.zero_outputs),
        ] {
            if wires.iter().any(|wire| *wire >= self.n)
                || wires.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(MatrixBoundaryError(format!(
                    "{name} must be sorted, unique, and inside the physical width"
                )));
            }
        }
        if self
            .protected_inputs
            .iter()
            .any(|wire| self.inputs.binary_search(wire).is_err())
        {
            return Err(MatrixBoundaryError(
                "protected wires must be selected inputs".into(),
            ));
        }
        if self
            .zero_outputs
            .iter()
            .any(|wire| self.outputs.binary_search(wire).is_err())
        {
            return Err(MatrixBoundaryError(
                "pinned-zero wires must be selected outputs".into(),
            ));
        }
        if self
            .weights
            .as_ref()
            .is_some_and(|weights| weights.len() != self.outputs.len())
        {
            return Err(MatrixBoundaryError(
                "one exact rational weight pair per output is required".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct BoundaryCacheEntry {
    state: StateIdentity,
    max_bytes: Option<usize>,
    values: HashMap<FeatureId, FeatureComputation>,
}

struct MatrixBoundaryContext {
    specs: Vec<MatrixBoundarySpec>,
    max_ports: usize,
    max_equations: usize,
    cache: Mutex<VecDeque<BoundaryCacheEntry>>,
}

impl MatrixBoundaryContext {
    fn values_for(
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

fn words_for_bits(bits: usize) -> usize {
    bits.saturating_add(63) / 64
}

fn bit_is_set(words: &[u64], bit: usize) -> bool {
    words
        .get(bit / 64)
        .is_some_and(|word| (word & (1u64 << (bit % 64))) != 0)
}

fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

fn matrix_bit(words: &[u64], bit: usize) -> bool {
    bit_is_set(words, bit)
}

fn xor_rows(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= *right;
    }
}

fn canonicalize_rows(mut rows: Vec<Vec<u64>>, bits: usize) -> Vec<Vec<u64>> {
    let mut pivot = 0usize;
    for bit in (0..bits).rev() {
        let Some(selected) = (pivot..rows.len()).find(|index| bit_is_set(&rows[*index], bit))
        else {
            continue;
        };
        rows.swap(pivot, selected);
        let pivot_row = rows[pivot].clone();
        for (index, row) in rows.iter_mut().enumerate() {
            if index != pivot && bit_is_set(row, bit) {
                xor_rows(row, &pivot_row);
            }
        }
        pivot += 1;
    }
    rows.into_iter()
        .filter(|row| row.iter().any(|word| *word != 0))
        .collect()
}

fn rank_binary_rows(rows: Vec<Vec<u64>>, bits: usize) -> usize {
    canonicalize_rows(rows, bits).len()
}

fn protected_projection_rank(
    spec: &MatrixBoundarySpec,
    relation_rows: &[Vec<u64>],
    ports: usize,
) -> usize {
    let protected_ports = spec
        .protected_inputs
        .iter()
        .filter_map(|wire| spec.inputs.binary_search(wire).ok())
        .collect::<Vec<_>>();
    let protected_port_set = protected_ports
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();

    let mut rows = relation_rows.to_vec();
    let mut pivot_count = 0usize;
    for bit in 0..ports {
        if bit < spec.inputs.len() && protected_port_set.contains(&bit) {
            continue;
        }
        let Some(selected) = (pivot_count..rows.len()).find(|index| bit_is_set(&rows[*index], bit))
        else {
            continue;
        };
        rows.swap(pivot_count, selected);
        let pivot_row = rows[pivot_count].clone();
        for (index, row) in rows.iter_mut().enumerate() {
            if index != pivot_count && bit_is_set(row, bit) {
                xor_rows(row, &pivot_row);
            }
        }
        pivot_count += 1;
    }

    let mut projected = Vec::new();
    for row in rows.iter().skip(pivot_count) {
        let mut output = vec![0u64; words_for_bits(protected_ports.len())];
        for (new_index, old_index) in protected_ports.iter().enumerate() {
            if bit_is_set(row, *old_index) {
                set_bit(&mut output, new_index);
            }
        }
        projected.push(output);
    }
    canonicalize_rows(projected, protected_ports.len()).len()
}

fn producer_version(
    specs: &[MatrixBoundarySpec],
    max_ports: usize,
    max_equations: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"selected_matrix_boundary_v1");
    hash_field(&mut digest, &(max_ports as u64).to_le_bytes());
    hash_field(&mut digest, &(max_equations as u64).to_le_bytes());
    for spec in specs {
        hash_field(&mut digest, &spec.n.to_le_bytes());
        for wires in [
            &spec.inputs,
            &spec.outputs,
            &spec.protected_inputs,
            &spec.zero_outputs,
        ] {
            hash_field(&mut digest, &(wires.len() as u64).to_le_bytes());
            for wire in wires {
                hash_field(&mut digest, &wire.to_le_bytes());
            }
        }
        if let Some(weights) = &spec.weights {
            hash_field(&mut digest, &[1]);
            hash_field(&mut digest, &(weights.len() as u64).to_le_bytes());
            for (left, right) in weights {
                for weight in [left, right] {
                    hash_field(&mut digest, weight.numerator.as_bytes());
                    hash_field(&mut digest, weight.denominator.as_bytes());
                }
            }
        } else {
            hash_field(&mut digest, &[0]);
        }
    }
    let hash = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("selected_matrix_boundary_v1:{hash}")
}

fn hash_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

struct MatrixBoundaryProducer {
    spec: FeatureSpec,
    id: FeatureId,
    context: Arc<MatrixBoundaryContext>,
}

impl FeatureProducer for MatrixBoundaryProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return FeatureComputation::Unavailable {
                status: FeatureStatus::Unsupported,
                reason: "matrix_boundary_requires_augmentation_only_symmetry".into(),
            };
        }
        self.context
            .values_for(state, policy.max_bytes)
            .remove(&self.id)
            .unwrap_or_else(|| FeatureComputation::Unavailable {
                status: FeatureStatus::Unsupported,
                reason: "matrix_boundary_provider_did_not_emit_feature".into(),
            })
    }
}

/// Register C08, I01-I04, and I11 together for explicit per-width boundaries.
/// Dynamic I01/I02 group shapes are declared by this provider, never guessed
/// from the static placeholder layout.
pub fn register_matrix_boundaries(
    registry: &mut FeatureRegistry,
    specs: Vec<MatrixBoundarySpec>,
    max_ports: usize,
    max_equations: usize,
) -> Result<(), MatrixBoundaryError> {
    if specs.is_empty() || max_ports == 0 || max_equations == 0 {
        return Err(MatrixBoundaryError(
            "explicit boundaries and finite positive port/equation budgets required".into(),
        ));
    }
    let mut widths = std::collections::HashSet::with_capacity(specs.len());
    for spec in &specs {
        spec.validate()?;
        if !widths.insert(spec.n) {
            return Err(MatrixBoundaryError(
                "at most one explicit boundary per matrix width is allowed".into(),
            ));
        }
    }

    let producer_version = producer_version(&specs, max_ports, max_equations);
    let context = Arc::new(MatrixBoundaryContext {
        specs,
        max_ports,
        max_equations,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    });
    let definitions = [
        ((b'C', 8), FeatureRole::Global, 1),
        ((b'I', 1), FeatureRole::Group, 2),
        ((b'I', 2), FeatureRole::Group, 3),
        ((b'I', 3), FeatureRole::Global, 2),
        ((b'I', 4), FeatureRole::Global, 3),
        ((b'I', 11), FeatureRole::Global, 4),
    ];
    for ((family, ordinal), role, channels) in definitions {
        let id = FeatureId::new(family, ordinal).expect("matrix-boundary ID is canonical");
        let mut spec = feature_spec(id);
        spec.role = role;
        spec.channels = channels;
        let wire_roles = if role == FeatureRole::Group {
            PORT_WIRE_ROLES.to_vec()
        } else {
            Vec::new()
        };
        let spec = spec.with_provider_metadata(
            "explicit_y=M[outputs,inputs]x;other_inputs_zero;named_ports;I03_projected_relation_flag_not_generic_zero;I04_positive_product_weights_zero_phase_only;state_counts_as_log2",
            producer_version.clone(),
            ["M", "physical_boundary_spec", "exact_affine_relation"],
            "exact",
            "T2",
            wire_roles,
            false,
        );
        registry.register(Box::new(MatrixBoundaryProducer {
            spec,
            id,
            context: Arc::clone(&context),
        }))?;
    }
    Ok(())
}
