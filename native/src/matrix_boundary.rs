//! Exact source-defined features for an explicitly selected linear boundary.
//!
//! This is the Rust port of research_v3.features.matrix_boundary. It models
//! y = M[outputs, inputs] x with all other inputs fixed to zero. It does not
//! infer a boundary from a matrix and does not claim a generic circuit cost.

use crate::features::FeatureBuildError;
use std::fmt;

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

mod context;
mod linear;
mod producer;

pub use producer::register_matrix_boundaries;
