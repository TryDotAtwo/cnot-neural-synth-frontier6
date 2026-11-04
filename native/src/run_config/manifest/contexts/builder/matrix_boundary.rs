//! Checked construction of rational matrix-boundary specifications.

use super::super::MatrixBoundarySpecInput;
use crate::matrix_boundary::{MatrixBoundarySpec, RationalWeight};

impl MatrixBoundarySpecInput {
    pub(crate) fn build(self) -> Result<MatrixBoundarySpec, String> {
        let mut spec = MatrixBoundarySpec::new(self.n, self.inputs, self.outputs)
            .map_err(|e| e.to_string())?
            .with_protected_inputs(self.protected_inputs)
            .map_err(|e| e.to_string())?
            .with_zero_outputs(self.zero_outputs)
            .map_err(|e| e.to_string())?;
        if !self.weights.is_empty() {
            let weights = self
                .weights
                .into_iter()
                .map(|[first, second]| {
                    Ok((
                        RationalWeight::new(first.numerator, first.denominator)
                            .map_err(|e| e.to_string())?,
                        RationalWeight::new(second.numerator, second.denominator)
                            .map_err(|e| e.to_string())?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            spec = spec.with_weights(weights).map_err(|e| e.to_string())?;
        }
        Ok(spec)
    }
}
