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
use std::collections::BTreeSet;
use std::fmt;

mod compiler;
mod summaries;
use compiler::{compile_equations, equation_bound, variable_count, working_bytes_upper_bound};
use summaries::{
    compute_alpha, compute_beta, equation_support, rank_beta, rank_mixed_terms, variable_names,
};

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
