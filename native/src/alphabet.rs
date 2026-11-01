//! Exact finite CNOT alphabets and replayable word certificates.

use crate::state::{CnotGate, MatrixError, MatrixState};
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlphabetError(pub String);

impl fmt::Display for AlphabetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AlphabetError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CnotAlphabet {
    dimension: u32,
    id: String,
    gates: Vec<CnotGate>,
    costs: Vec<u32>,
    indices: HashMap<CnotGate, usize>,
}

impl CnotAlphabet {
    pub fn new(
        dimension: u32,
        id: impl Into<String>,
        gates: Vec<CnotGate>,
        costs: Vec<u32>,
    ) -> Result<Self, AlphabetError> {
        if dimension == 0 {
            return Err(AlphabetError("alphabet dimension must be positive".into()));
        }
        let id = id.into();
        if id.trim().is_empty() {
            return Err(AlphabetError("alphabet identity must be nonempty".into()));
        }
        if gates.len() != costs.len() {
            return Err(AlphabetError("gate and cost lengths differ".into()));
        }
        let mut indices = HashMap::with_capacity(gates.len());
        for (index, (gate, cost)) in gates.iter().copied().zip(costs.iter().copied()).enumerate() {
            if gate.control >= dimension || gate.target >= dimension || gate.control == gate.target
            {
                return Err(AlphabetError(format!(
                    "gate {index} is outside the CNOT alphabet"
                )));
            }
            if cost == 0 {
                return Err(AlphabetError(format!(
                    "gate {index} has nonpositive physical cost"
                )));
            }
            if indices.insert(gate, index).is_some() {
                return Err(AlphabetError(format!(
                    "duplicate gate at alphabet index {index}"
                )));
            }
        }
        Ok(Self {
            dimension,
            id,
            gates,
            costs,
            indices,
        })
    }

    pub fn all_to_all_unit(dimension: u32) -> Result<Self, AlphabetError> {
        if dimension == 0 {
            return Err(AlphabetError("alphabet dimension must be positive".into()));
        }
        let action_count = usize::try_from(dimension)
            .ok()
            .and_then(|n| n.checked_mul(n.saturating_sub(1)))
            .ok_or_else(|| AlphabetError("alphabet size overflow".into()))?;
        let mut gates = Vec::with_capacity(action_count);
        for control in 0..dimension {
            for target in 0..dimension {
                if control != target {
                    gates.push(CnotGate { control, target });
                }
            }
        }
        Self::new(
            dimension,
            format!("all_to_all_unit_cnot:{dimension}"),
            gates,
            vec![1; action_count],
        )
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn gates(&self) -> &[CnotGate] {
        &self.gates
    }

    pub fn cost(&self, gate: CnotGate) -> Option<u32> {
        self.indices.get(&gate).map(|index| self.costs[*index])
    }

    pub fn cost_at(&self, index: usize) -> Option<u32> {
        self.costs.get(index).copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WordCertificate {
    pub start: MatrixState,
    pub end: MatrixState,
    pub gates: Vec<CnotGate>,
    pub physical_cost: u64,
    pub producer: String,
    pub lineage_id: String,
}

impl WordCertificate {
    /// Replays every gate under the project's row-target XOR row-control convention.
    /// This checks the supplied word and cost; it does not prove that the word is shortest.
    pub fn verify(&self, alphabet: &CnotAlphabet) -> Result<MatrixState, CertificateError> {
        if self.start.dimension() != alphabet.dimension()
            || self.end.dimension() != alphabet.dimension()
            || self.start.alphabet_id() != alphabet.id()
            || self.end.alphabet_id() != alphabet.id()
        {
            return Err(CertificateError(
                "certificate state does not match alphabet".into(),
            ));
        }
        if self.producer.trim().is_empty() || self.lineage_id.trim().is_empty() {
            return Err(CertificateError(
                "certificate producer and lineage are required".into(),
            ));
        }
        let mut state = self.start.clone();
        let mut actual_cost = 0u64;
        for (index, gate) in self.gates.iter().copied().enumerate() {
            let cost = alphabet.cost(gate).ok_or_else(|| {
                CertificateError(format!("gate {index} is not in the selected alphabet"))
            })?;
            actual_cost = actual_cost
                .checked_add(u64::from(cost))
                .ok_or_else(|| CertificateError("word cost overflow".into()))?;
            state.apply_cnot(gate).map_err(CertificateError::from)?;
        }
        if actual_cost != self.physical_cost {
            return Err(CertificateError(format!(
                "certificate cost mismatch: declared {}, replayed {actual_cost}",
                self.physical_cost
            )));
        }
        if state != self.end {
            return Err(CertificateError(
                "replayed word does not reach the declared endpoint".into(),
            ));
        }
        Ok(state)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertificateError(pub String);

impl fmt::Display for CertificateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CertificateError {}

impl From<MatrixError> for CertificateError {
    fn from(error: MatrixError) -> Self {
        Self(error.to_string())
    }
}
