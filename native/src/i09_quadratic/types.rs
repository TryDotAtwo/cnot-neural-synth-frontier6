use super::compiler::variable_count;
use super::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_EQUATIONS, DEFAULT_MAX_VARIABLES, I09_STAGE60_SOURCE_SHA256,
};
use sha2::{Digest, Sha256};
use std::fmt;

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
    pub(super) fn new(kind: I09ErrorKind, reason: impl Into<String>) -> Self {
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
    pub(super) n: usize,
    pub(super) horizon: usize,
    pub(super) cut: usize,
    pub(super) prefix_bits: Vec<u8>,
    pub(super) allowed_gates: Vec<(usize, usize)>,
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

pub(super) fn context_digest(context: &I09Context) -> String {
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
