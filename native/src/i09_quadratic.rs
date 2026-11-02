//! Exact, resource-bounded Rust port of the pinned Stage-60 I09 response.
//!
//! This provider emits the compiler equations and the source's fixed-prefix
//! alpha/beta/rank summaries. It does not execute a solver or claim general
//! Frontier61 acceptance.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::mem::size_of;

/// SHA-256 of the separately pinned Stage-60 source used by the I09 reference.
pub const I09_STAGE60_SOURCE_SHA256: &str =
    "fcdaec83bff7c111261dfc46c2040ddbbc349dbbc11a712a31af945dca3c6339";

const DEFAULT_MAX_VARIABLES: usize = 128;
const DEFAULT_MAX_EQUATIONS: usize = 512;
const DEFAULT_MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct I09ContextError(pub String);

impl fmt::Display for I09ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for I09ContextError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum I09ErrorKind {
    Unsupported,
    ResourceLimit,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct I09Error {
    pub kind: I09ErrorKind,
    pub reason: String,
}

impl I09Error {
    fn new(kind: I09ErrorKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for I09Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for I09Error {}

/// A fixed-horizon optional-CNOT compiler instance and its labelled prefix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct I09Context {
    n: usize,
    horizon: usize,
    cut: usize,
    prefix_bits: Vec<u8>,
    allowed_gates: Vec<(usize, usize)>,
}

impl I09Context {
    pub fn new(
        n: usize,
        horizon: usize,
        cut: usize,
        prefix_bits: Vec<u8>,
        allowed_gates: Vec<(usize, usize)>,
    ) -> Result<Self, I09ContextError> {
        if n == 0 {
            return Err(I09ContextError(
                "I09 requires a positive matrix width".into(),
            ));
        }
        let variable_count = variable_count(n, horizon).ok_or_else(|| {
            I09ContextError("I09 variable count overflows the platform size".into())
        })?;
        if cut > variable_count || prefix_bits.len() != cut {
            return Err(I09ContextError(
                "I09 prefix length must equal a cut within the variable order".into(),
            ));
        }
        if prefix_bits.iter().any(|bit| *bit > 1) {
            return Err(I09ContextError(
                "I09 prefix bits must be zero or one".into(),
            ));
        }
        if allowed_gates
            .iter()
            .any(|(control, target)| *control >= n || *target >= n || control == target)
        {
            return Err(I09ContextError(
                "I09 allowed gates must be valid directed CNOTs".into(),
            ));
        }
        if allowed_gates.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(I09ContextError(
                "I09 allowed gates must be sorted and unique".into(),
            ));
        }
        Ok(Self {
            n,
            horizon,
            cut,
            prefix_bits,
            allowed_gates,
        })
    }

    pub fn width(&self) -> usize {
        self.n
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }

    pub fn cut(&self) -> usize {
        self.cut
    }

    pub fn prefix_bits(&self) -> &[u8] {
        &self.prefix_bits
    }

    pub fn allowed_gates(&self) -> &[(usize, usize)] {
        &self.allowed_gates
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct I09Limits {
    pub max_variables: usize,
    pub max_equations: usize,
    pub max_bytes: usize,
}

impl Default for I09Limits {
    fn default() -> Self {
        Self {
            max_variables: DEFAULT_MAX_VARIABLES,
            max_equations: DEFAULT_MAX_EQUATIONS,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// Canonical Boolean monomials. Products are emitted with ordered indices;
/// duplicate variables reduce by the Boolean identity x*x = x.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum I09Monomial {
    Constant,
    Variable(usize),
    Product(usize, usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct I09Response {
    pub variable_names: Vec<String>,
    /// Each equation is a GF(2) sum of canonical Boolean monomials, equal to 0.
    pub equations: Vec<Vec<I09Monomial>>,
    pub alpha: Vec<u8>,
    /// Equation-major beta rows; columns correspond to variables after `cut`.
    pub beta: Vec<Vec<u8>>,
    pub rank_b: usize,
    pub mixed_rank: usize,
    /// Equation-major physical support, with one `equation_physical_support` role.
    pub equation_support: Vec<u8>,
}

#[derive(Clone)]
struct I09Producer {
    spec: FeatureSpec,
    contexts: Vec<I09Context>,
    limits: I09Limits,
}

impl FeatureProducer for I09Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(FeatureStatus::Unsupported, "i09_requires_augmentation_only");
        }
        let Some(context) = self
            .contexts
            .iter()
            .find(|context| context.n == state.dimension() as usize)
        else {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_explicit_i09_context_for_matrix_width",
            );
        };
        let expected_alphabet = format!("all_to_all_unit_cnot:{}", context.n);
        if state.alphabet_id() != expected_alphabet {
            return unavailable(
                FeatureStatus::Unsupported,
                "i09_requires_matching_all_to_all_unit_cnot_alphabet",
            );
        }
        match compute_i09_response(state, context, self.limits) {
            Ok(response) => match response.to_feature_value(self.limits, context.n) {
                Ok(value) => FeatureComputation::ComputedWithProvenance {
                    value,
                    provenance: vec![format!(
                        "stage60_source_sha256={};context_sha256={}",
                        I09_STAGE60_SOURCE_SHA256,
                        context_digest(context)
                    )],
                },
                Err(error) => unavailable(
                    FeatureStatus::ResourceLimit,
                    &format!("i09_output_conversion_failed:{}", error.reason),
                ),
            },
            Err(error) => unavailable(
                match error.kind {
                    I09ErrorKind::Unsupported => FeatureStatus::Unsupported,
                    I09ErrorKind::ResourceLimit | I09ErrorKind::Internal => {
                        FeatureStatus::ResourceLimit
                    }
                },
                &error.reason,
            ),
        }
    }
}

impl I09Response {
    fn to_feature_value(&self, limits: I09Limits, n: usize) -> Result<FeatureValue, I09Error> {
        let groups = self.equations.len();
        let channels = limits
            .max_variables
            .checked_add(4)
            .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_channel_overflow"))?;
        let value_count = groups.checked_mul(channels).ok_or_else(|| {
            I09Error::new(I09ErrorKind::ResourceLimit, "i09_output_size_overflow")
        })?;
        if groups > MAX_EXACT_F32_INTEGER || channels > MAX_EXACT_F32_INTEGER {
            return Err(I09Error::new(
                I09ErrorKind::ResourceLimit,
                "i09_float_labels_exceed_exact_integer_range",
            ));
        }
        let mut values = Vec::new();
        values.try_reserve_exact(value_count).map_err(|_| {
            I09Error::new(I09ErrorKind::ResourceLimit, "i09_output_allocation_failed")
        })?;
        values.resize(value_count, 0.0f32);
        let mut incidence = Vec::new();
        incidence
            .try_reserve_exact(groups.checked_mul(n).ok_or_else(|| {
                I09Error::new(I09ErrorKind::ResourceLimit, "i09_incidence_size_overflow")
            })?)
            .map_err(|_| {
                I09Error::new(
                    I09ErrorKind::ResourceLimit,
                    "i09_incidence_allocation_failed",
                )
            })?;
        for equation_index in 0..groups {
            let base = equation_index * channels;
            values[base] = equation_index as f32;
            values[base + 1] = f32::from(self.alpha[equation_index]);
            values[base + 2] = self.rank_b as f32;
            values[base + 3] = self.mixed_rank as f32;
            for (future, bit) in self.beta[equation_index].iter().copied().enumerate() {
                if future < limits.max_variables {
                    values[base + 4 + future] = f32::from(bit);
                }
            }
            let support_start = equation_index * n;
            incidence.extend(
                self.equation_support[support_start..support_start + n]
                    .iter()
                    .map(|value| f32::from(*value)),
            );
        }
        Ok(FeatureValue {
            shape: vec![groups, channels],
            values,
            token_valid: vec![1; groups],
            incidence,
        })
    }
}

/// Compile the bounded optional-CNOT equations and exact Stage-60 summaries.
pub fn compute_i09_response(
    state: &MatrixState,
    context: &I09Context,
    limits: I09Limits,
) -> Result<I09Response, I09Error> {
    if state.dimension() as usize != context.n {
        return Err(I09Error::new(
            I09ErrorKind::Unsupported,
            "i09_state_width_does_not_match_context",
        ));
    }
    let expected_alphabet = format!("all_to_all_unit_cnot:{}", context.n);
    if state.alphabet_id() != expected_alphabet {
        return Err(I09Error::new(
            I09ErrorKind::Unsupported,
            "i09_requires_matching_all_to_all_unit_cnot_alphabet",
        ));
    }
    let variables = variable_count(context.n, context.horizon)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_variable_count_overflow"))?;
    let equation_bound = equation_bound(context.n, context.horizon)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_equation_bound_overflow"))?;
    if limits.max_variables == 0
        || limits.max_equations == 0
        || limits.max_bytes == 0
        || variables > limits.max_variables
        || equation_bound > limits.max_equations
        || variables > MAX_EXACT_F32_INTEGER
        || equation_bound > MAX_EXACT_F32_INTEGER
    {
        return Err(I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_context_exceeds_declared_variable_equation_or_label_budget",
        ));
    }
    let estimate = working_bytes_upper_bound(context.n, context.horizon, variables, equation_bound)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_working_size_overflow"))?;
    if estimate > limits.max_bytes {
        return Err(I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_working_memory_budget_exceeded_before_allocation",
        ));
    }

    let equations = compile_equations(state, context, variables)?;
    if equations.len() > limits.max_equations {
        return Err(I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_compiled_equation_count_exceeded_declared_budget",
        ));
    }
    let alpha = compute_alpha(&equations, context);
    let beta = compute_beta(&equations, context, variables)?;
    let rank_b = rank_beta(&beta, equations.len())?;
    let mixed_rank = rank_mixed_terms(&equations, context, variables)?;
    let equation_support = equation_support(&equations, context.n, context.horizon, variables)?;
    let variable_names = variable_names(context.n, context.horizon, variables)?;
    Ok(I09Response {
        variable_names,
        equations,
        alpha,
        beta,
        rank_b,
        mixed_rank,
        equation_support,
    })
}

/// Registers I09 for explicit width-labelled Stage-60 compiler contexts.
pub fn register_i09_quadratic_response(
    registry: &mut FeatureRegistry,
    contexts: Vec<I09Context>,
    limits: I09Limits,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty()
        || limits.max_variables == 0
        || limits.max_equations == 0
        || limits.max_bytes == 0
    {
        return Err(FeatureBuildError(
            "I09 requires explicit contexts and positive resource limits".into(),
        ));
    }
    let mut widths = BTreeSet::new();
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I09 permits one explicit context per matrix width".into(),
        ));
    }
    if limits.max_variables > MAX_EXACT_F32_INTEGER || limits.max_equations > MAX_EXACT_F32_INTEGER
    {
        return Err(FeatureBuildError(
            "I09 float labels must remain exact integers".into(),
        ));
    }
    let id = FeatureId::new(b'I', 9).expect("I09 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = limits
        .max_variables
        .checked_add(4)
        .ok_or_else(|| FeatureBuildError("I09 channel count overflows the platform size".into()))?;
    let version = producer_version(&contexts, limits);
    let spec = spec.with_provider_metadata(
        "pinned_stage60_optional_cnot_quadratic_equations;alpha,beta,rank_B,mixed_rank;not_solver_or_distance",
        version,
        ["M", "explicit_optional_cnot_context", "stage60_source_sha256"],
        "exact_within_declared_context_and_resource_limits",
        "T2",
        ["equation_physical_support"],
        false,
    );
    registry.register(Box::new(I09Producer {
        spec,
        contexts,
        limits,
    }))
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn variable_count(n: usize, horizon: usize) -> Option<usize> {
    let per_slot = n
        .checked_mul(n)?
        .checked_add(n.checked_mul(3)?)?
        .checked_add(1)?;
    horizon.checked_mul(per_slot)
}

fn equation_bound(n: usize, horizon: usize) -> Option<usize> {
    let n_squared = n.checked_mul(n)?;
    let per_slot = 3usize
        .checked_add(n.checked_mul(n.checked_sub(1)?)?)?
        .checked_add(n_squared.checked_mul(2)?)?
        .checked_add(n)?;
    horizon.checked_mul(per_slot)?.checked_add(n_squared)
}

fn working_bytes_upper_bound(
    n: usize,
    horizon: usize,
    variables: usize,
    equations: usize,
) -> Option<usize> {
    let channels = variables.checked_add(4)?;
    let feature_bytes = equations
        .checked_mul(channels.checked_add(n)?)?
        .checked_mul(size_of::<f32>())?;
    let terms = equations.checked_mul(n.checked_add(2)?)?;
    let equation_bytes = terms
        .checked_mul(size_of::<I09Monomial>())?
        .checked_add(equations.checked_mul(size_of::<Vec<I09Monomial>>())?)?;
    let mixed_bits = equations.checked_mul(variables)?;
    let mixed_words = mixed_bits.checked_add(63)?.checked_div(64)?;
    let rank_bytes = variables
        .checked_mul(mixed_words)?
        .checked_mul(size_of::<u64>())?
        .checked_add(
            variables
                .checked_mul(equations.checked_add(63)?.checked_div(64)?)?
                .checked_mul(size_of::<u64>())?,
        )?;
    let metadata_bytes = variables
        .checked_mul(64)?
        .checked_add(equations.checked_mul(size_of::<u8>())?)?
        .checked_add(n.checked_mul(horizon)?)?;
    feature_bytes
        .checked_add(equation_bytes)?
        .checked_add(rank_bytes)?
        .checked_add(metadata_bytes)
}

fn slot_width(n: usize) -> Option<usize> {
    n.checked_mul(n)?
        .checked_add(n.checked_mul(3)?)?
        .checked_add(1)
}

fn active_var(slot: usize, n: usize) -> Option<usize> {
    slot.checked_mul(slot_width(n)?)
}

fn control_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    active_var(slot, n)?.checked_add(1)?.checked_add(wire)
}

fn target_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    control_var(slot, n, n)?.checked_add(wire)
}

fn readout_var(slot: usize, wire: usize, n: usize) -> Option<usize> {
    control_var(slot, n.checked_mul(2)?, n)?.checked_add(wire)
}

fn matrix_var(slot: usize, row: usize, column: usize, n: usize) -> Option<usize> {
    let offset = row.checked_mul(n)?.checked_add(column)?;
    readout_var(slot, n, n)?.checked_add(offset)
}

fn compile_equations(
    state: &MatrixState,
    context: &I09Context,
    variable_count: usize,
) -> Result<Vec<Vec<I09Monomial>>, I09Error> {
    let n = context.n;
    let mut equations = Vec::new();
    equations
        .try_reserve_exact(equation_bound(n, context.horizon).unwrap_or_default())
        .map_err(|_| {
            I09Error::new(
                I09ErrorKind::ResourceLimit,
                "i09_equation_allocation_failed",
            )
        })?;
    for slot in 0..context.horizon {
        let active = checked_index(active_var(slot, n))?;
        let mut active_controls = Polynomial::new();
        active_controls.toggle(I09Monomial::Variable(active));
        let mut active_targets = Polynomial::new();
        active_targets.toggle(I09Monomial::Variable(active));
        for wire in 0..n {
            active_controls.toggle(I09Monomial::Variable(checked_index(control_var(
                slot, wire, n,
            ))?));
            active_targets.toggle(I09Monomial::Variable(checked_index(target_var(
                slot, wire, n,
            ))?));
        }
        equations.push(active_controls.finish());
        equations.push(active_targets.finish());

        for left in 0..n {
            for right in left + 1..n {
                equations.push(vec![product(
                    checked_index(control_var(slot, left, n))?,
                    checked_index(control_var(slot, right, n))?,
                )]);
            }
        }
        for left in 0..n {
            for right in left + 1..n {
                equations.push(vec![product(
                    checked_index(target_var(slot, left, n))?,
                    checked_index(target_var(slot, right, n))?,
                )]);
            }
        }
        for control in 0..n {
            for target in 0..n {
                if !context.allowed_gates.contains(&(control, target)) {
                    equations.push(vec![product(
                        checked_index(control_var(slot, control, n))?,
                        checked_index(target_var(slot, target, n))?,
                    )]);
                }
            }
        }
        if slot > 0 {
            let mut monotone = Polynomial::new();
            let current = checked_index(active_var(slot, n))?;
            let previous = checked_index(active_var(slot - 1, n))?;
            monotone.toggle(I09Monomial::Variable(current));
            monotone.toggle(product(previous, current));
            equations.push(monotone.finish());
        }
        for column in 0..n {
            let mut readout = Polynomial::new();
            readout.toggle(I09Monomial::Variable(checked_index(readout_var(
                slot, column, n,
            ))?));
            for row in 0..n {
                let control = checked_index(control_var(slot, row, n))?;
                if slot == 0 {
                    if row == column {
                        readout.toggle(I09Monomial::Variable(control));
                    }
                } else {
                    readout.toggle(product(
                        control,
                        checked_index(matrix_var(slot - 1, row, column, n))?,
                    ));
                }
            }
            equations.push(readout.finish());
        }
        for row in 0..n {
            for column in 0..n {
                let mut update = Polynomial::new();
                update.toggle(I09Monomial::Variable(checked_index(matrix_var(
                    slot, row, column, n,
                ))?));
                if slot == 0 {
                    if row == column {
                        update.toggle(I09Monomial::Constant);
                    }
                } else {
                    update.toggle(I09Monomial::Variable(checked_index(matrix_var(
                        slot - 1,
                        row,
                        column,
                        n,
                    ))?));
                }
                update.toggle(product(
                    checked_index(target_var(slot, row, n))?,
                    checked_index(readout_var(slot, column, n))?,
                ));
                equations.push(update.finish());
            }
        }
    }
    let last_slot = context.horizon.checked_sub(1);
    for row in 0..n {
        for column in 0..n {
            let mut boundary = Polynomial::new();
            let target_bit = state
                .row_words(row as u32)
                .and_then(|words| words.get(column / 64))
                .is_some_and(|word| ((word >> (column % 64)) & 1) != 0);
            if let Some(last_slot) = last_slot {
                boundary.toggle(I09Monomial::Variable(checked_index(matrix_var(
                    last_slot, row, column, n,
                ))?));
                if target_bit {
                    boundary.toggle(I09Monomial::Constant);
                }
            } else if (row == column) != target_bit {
                // With no CNOT slots, the compiler starts and ends at I.
                boundary.toggle(I09Monomial::Constant);
            }
            equations.push(boundary.finish());
        }
    }
    if equations
        .iter()
        .flat_map(|equation| equation.iter())
        .any(|term| !monomial_is_in_range(*term, variable_count))
    {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_compiler_emitted_an_out_of_range_variable",
        ));
    }
    Ok(equations)
}

#[derive(Default)]
struct Polynomial(BTreeSet<I09Monomial>);

impl Polynomial {
    fn new() -> Self {
        Self::default()
    }

    fn toggle(&mut self, term: I09Monomial) {
        if !self.0.insert(term) {
            self.0.remove(&term);
        }
    }

    fn finish(self) -> Vec<I09Monomial> {
        self.0.into_iter().collect()
    }
}

fn product(left: usize, right: usize) -> I09Monomial {
    if left == right {
        I09Monomial::Variable(left)
    } else if left < right {
        I09Monomial::Product(left, right)
    } else {
        I09Monomial::Product(right, left)
    }
}

fn checked_index(index: Option<usize>) -> Result<usize, I09Error> {
    index.ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_variable_index_overflow"))
}

fn monomial_is_in_range(term: I09Monomial, variable_count: usize) -> bool {
    match term {
        I09Monomial::Constant => true,
        I09Monomial::Variable(index) => index < variable_count,
        I09Monomial::Product(left, right) => left < variable_count && right < variable_count,
    }
}

fn compute_alpha(equations: &[Vec<I09Monomial>], context: &I09Context) -> Vec<u8> {
    equations
        .iter()
        .map(|equation| {
            equation.iter().fold(0u8, |value, term| {
                let contribution = match *term {
                    I09Monomial::Constant => Some(1),
                    I09Monomial::Variable(index) if index < context.cut => {
                        Some(context.prefix_bits[index])
                    }
                    I09Monomial::Variable(_) => None,
                    I09Monomial::Product(left, right)
                        if left < context.cut && right < context.cut =>
                    {
                        Some(context.prefix_bits[left] & context.prefix_bits[right])
                    }
                    I09Monomial::Product(_, _) => None,
                };
                value ^ contribution.unwrap_or(0)
            })
        })
        .collect()
}

fn compute_beta(
    equations: &[Vec<I09Monomial>],
    context: &I09Context,
    variables: usize,
) -> Result<Vec<Vec<u8>>, I09Error> {
    let future_count = variables
        .checked_sub(context.cut)
        .ok_or_else(|| I09Error::new(I09ErrorKind::Internal, "i09_cut_exceeds_variable_order"))?;
    let mut beta = Vec::new();
    beta.try_reserve_exact(equations.len())
        .map_err(|_| I09Error::new(I09ErrorKind::ResourceLimit, "i09_beta_allocation_failed"))?;
    for equation in equations {
        let mut row = vec![0u8; future_count];
        for term in equation {
            let I09Monomial::Product(left, right) = *term else {
                continue;
            };
            let pair = if left < context.cut && right >= context.cut {
                Some((left, right))
            } else if right < context.cut && left >= context.cut {
                Some((right, left))
            } else {
                None
            };
            if let Some((past, future)) = pair {
                if context.prefix_bits[past] == 1 {
                    let index = future.checked_sub(context.cut).ok_or_else(|| {
                        I09Error::new(I09ErrorKind::Internal, "i09_beta_future_index_underflow")
                    })?;
                    if index >= row.len() {
                        return Err(I09Error::new(
                            I09ErrorKind::Internal,
                            "i09_beta_future_index_out_of_range",
                        ));
                    }
                    row[index] ^= 1;
                }
            }
        }
        beta.push(row);
    }
    Ok(beta)
}

fn rank_beta(beta: &[Vec<u8>], equation_count: usize) -> Result<usize, I09Error> {
    let columns = beta.first().map_or(0, Vec::len);
    let words = equation_count.checked_add(63).ok_or_else(|| {
        I09Error::new(I09ErrorKind::ResourceLimit, "i09_rank_word_count_overflow")
    })? / 64;
    let mut vectors = Vec::new();
    vectors.try_reserve_exact(columns).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_beta_rank_allocation_failed",
        )
    })?;
    for column in 0..columns {
        let mut vector = vec![0u64; words];
        for (equation, row) in beta.iter().enumerate() {
            if row.get(column) == Some(&1) {
                vector[equation / 64] |= 1u64 << (equation % 64);
            }
        }
        vectors.push(vector);
    }
    gf2_rank(vectors, equation_count)
}

fn rank_mixed_terms(
    equations: &[Vec<I09Monomial>],
    context: &I09Context,
    variables: usize,
) -> Result<usize, I09Error> {
    let future_count = variables
        .checked_sub(context.cut)
        .ok_or_else(|| I09Error::new(I09ErrorKind::Internal, "i09_cut_exceeds_variable_order"))?;
    let bit_count = equations.len().checked_mul(future_count).ok_or_else(|| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_mixed_rank_bit_count_overflow",
        )
    })?;
    let words = bit_count.checked_add(63).ok_or_else(|| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_mixed_rank_word_count_overflow",
        )
    })? / 64;
    let mut vectors = Vec::new();
    vectors.try_reserve_exact(context.cut).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_mixed_rank_allocation_failed",
        )
    })?;
    for _ in 0..context.cut {
        vectors.push(vec![0u64; words]);
    }
    for (equation_index, equation) in equations.iter().enumerate() {
        for term in equation {
            let I09Monomial::Product(left, right) = *term else {
                continue;
            };
            let pair = if left < context.cut && right >= context.cut {
                Some((left, right))
            } else if right < context.cut && left >= context.cut {
                Some((right, left))
            } else {
                None
            };
            if let Some((past, future)) = pair {
                let future_index = future - context.cut;
                let bit = equation_index
                    .checked_mul(future_count)
                    .and_then(|base| base.checked_add(future_index))
                    .ok_or_else(|| {
                        I09Error::new(I09ErrorKind::ResourceLimit, "i09_mixed_bit_index_overflow")
                    })?;
                vectors[past][bit / 64] ^= 1u64 << (bit % 64);
            }
        }
    }
    gf2_rank(vectors, bit_count)
}

fn gf2_rank(vectors: Vec<Vec<u64>>, bit_count: usize) -> Result<usize, I09Error> {
    let words = bit_count.checked_add(63).ok_or_else(|| {
        I09Error::new(I09ErrorKind::ResourceLimit, "i09_rank_word_count_overflow")
    })? / 64;
    if vectors.iter().any(|vector| vector.len() != words) {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_rank_vector_has_wrong_width",
        ));
    }
    let mut pivots: HashMap<usize, Vec<u64>> = HashMap::new();
    pivots
        .try_reserve(vectors.len().min(bit_count))
        .map_err(|_| {
            I09Error::new(
                I09ErrorKind::ResourceLimit,
                "i09_rank_pivot_allocation_failed",
            )
        })?;
    for mut vector in vectors {
        for pivot_bit in (0..bit_count).rev() {
            if vector[pivot_bit / 64] & (1u64 << (pivot_bit % 64)) == 0 {
                continue;
            }
            if let Some(pivot) = pivots.get(&pivot_bit) {
                for (word, basis_word) in vector.iter_mut().zip(pivot) {
                    *word ^= basis_word;
                }
            } else {
                pivots.insert(pivot_bit, vector);
                break;
            }
        }
    }
    Ok(pivots.len())
}

fn equation_support(
    equations: &[Vec<I09Monomial>],
    n: usize,
    horizon: usize,
    variables: usize,
) -> Result<Vec<u8>, I09Error> {
    let support_len = equations
        .len()
        .checked_mul(n)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_support_size_overflow"))?;
    let mut support = vec![0u8; support_len];
    for (equation_index, equation) in equations.iter().enumerate() {
        for term in equation {
            match *term {
                I09Monomial::Constant => {}
                I09Monomial::Variable(index) => add_variable_support(
                    &mut support,
                    equation_index,
                    index,
                    n,
                    horizon,
                    variables,
                )?,
                I09Monomial::Product(left, right) => {
                    add_variable_support(
                        &mut support,
                        equation_index,
                        left,
                        n,
                        horizon,
                        variables,
                    )?;
                    add_variable_support(
                        &mut support,
                        equation_index,
                        right,
                        n,
                        horizon,
                        variables,
                    )?;
                }
            }
        }
    }
    Ok(support)
}

fn add_variable_support(
    support: &mut [u8],
    equation: usize,
    index: usize,
    n: usize,
    horizon: usize,
    variables: usize,
) -> Result<(), I09Error> {
    if index >= variables || index / slot_width(n).unwrap_or(0) >= horizon {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_support_variable_index_out_of_range",
        ));
    }
    let slot_size = slot_width(n)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_slot_width_overflow"))?;
    let within = index % slot_size;
    let base = equation
        .checked_mul(n)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_support_index_overflow"))?;
    if within == 0 {
        support[base..base + n].fill(1);
    } else if within <= n {
        support[base + within - 1] = 1;
    } else if within <= 2 * n {
        support[base + within - n - 1] = 1;
    } else if within <= 3 * n {
        support[base + within - 2 * n - 1] = 1;
    } else {
        let offset = within - (1 + 3 * n);
        let row = offset / n;
        let column = offset % n;
        support[base + row] = 1;
        support[base + column] = 1;
    }
    Ok(())
}

fn variable_names(n: usize, horizon: usize, variables: usize) -> Result<Vec<String>, I09Error> {
    let mut names = Vec::new();
    names.try_reserve_exact(variables).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_variable_names_allocation_failed",
        )
    })?;
    for slot in 0..horizon {
        names.push(format!("active[{slot}]"));
        for wire in 0..n {
            names.push(format!("control[{slot},{wire}]"));
        }
        for wire in 0..n {
            names.push(format!("target[{slot},{wire}]"));
        }
        for wire in 0..n {
            names.push(format!("readout[{slot},{wire}]"));
        }
        for row in 0..n {
            for column in 0..n {
                names.push(format!("matrix[{slot},{row},{column}]"));
            }
        }
    }
    if names.len() != variables {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_variable_name_count_mismatch",
        ));
    }
    Ok(names)
}

fn context_digest(context: &I09Context) -> String {
    let mut digest = Sha256::new();
    digest.update(I09_STAGE60_SOURCE_SHA256.as_bytes());
    digest.update((context.n as u64).to_le_bytes());
    digest.update((context.horizon as u64).to_le_bytes());
    digest.update((context.cut as u64).to_le_bytes());
    digest.update((context.prefix_bits.len() as u64).to_le_bytes());
    digest.update(&context.prefix_bits);
    digest.update((context.allowed_gates.len() as u64).to_le_bytes());
    for (control, target) in &context.allowed_gates {
        digest.update((*control as u64).to_le_bytes());
        digest.update((*target as u64).to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn producer_version(contexts: &[I09Context], limits: I09Limits) -> String {
    let mut digest = Sha256::new();
    digest.update(b"i09_stage60_quadratic_response_rust_v1");
    digest.update(I09_STAGE60_SOURCE_SHA256.as_bytes());
    digest.update((limits.max_variables as u64).to_le_bytes());
    digest.update((limits.max_equations as u64).to_le_bytes());
    digest.update((limits.max_bytes as u64).to_le_bytes());
    for context in contexts {
        digest.update(context_digest(context).as_bytes());
    }
    format!("i09_stage60_rust_v1:{:x}", digest.finalize())
}

/// Helper for consumers that need to assert the pinned source at runtime.
pub fn i09_source_sha256() -> &'static str {
    I09_STAGE60_SOURCE_SHA256
}
