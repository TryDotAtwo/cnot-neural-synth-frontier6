//! Construction of alphabet and gate specifications.

use super::super::{AlphabetSpec, GateSpec};
use crate::alphabet::CnotAlphabet;
use crate::CnotGate;

impl GateSpec {
    pub(crate) fn build(self) -> CnotGate {
        CnotGate {
            control: self.control,
            target: self.target,
        }
    }
}

impl AlphabetSpec {
    pub(crate) fn build(&self) -> Result<CnotAlphabet, String> {
        CnotAlphabet::new(
            self.dimension,
            self.id.clone(),
            self.gates.iter().copied().map(GateSpec::build).collect(),
            self.costs.clone(),
        )
        .map_err(|e| e.to_string())
    }
}
