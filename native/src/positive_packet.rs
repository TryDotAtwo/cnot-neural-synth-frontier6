//! Exact I08 port of the pinned stage-59 affine optional-CNOT packet.
//!
//! The feature describes cut widths and still-live selector labels. It is not
//! a distance oracle and does not silently linearize nonlinear endpoint maps.

use num_bigint::BigUint;
use std::collections::HashSet;
use std::fmt;

/// SHA-256 of the stage-59 source used by the research-v3 I08 implementation.
pub const STAGE59_SOURCE_SHA256: &str =
    "63824b056fa1983a10b3d535532c2975c7c3a1c96cfb2a90267ead2a1b168987";

const DEFAULT_MAX_N: usize = 144;
const DEFAULT_MAX_SLOTS: usize = 256;
const DEFAULT_MAX_MONOMIALS: usize = 4096;
const DEFAULT_MAX_FACTOR_ENTRIES: usize = 65_536;
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositivePacketContextError(pub String);

impl fmt::Display for PositivePacketContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PositivePacketContextError {}

/// One Boolean tensor factor over ordered selector identities.
///
/// Bit `j` of each allowed `BigUint` labels the assignment of `scope[j]`.
/// Scope order is significant and is preserved in the producer identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalBooleanFactor {
    scope: Vec<u32>,
    allowed: Vec<BigUint>,
}

impl LocalBooleanFactor {
    pub fn new(scope: Vec<u32>, allowed: Vec<BigUint>) -> Result<Self, PositivePacketContextError> {
        let mut seen = HashSet::with_capacity(scope.len());
        if scope.iter().any(|index| !seen.insert(*index)) {
            return Err(PositivePacketContextError(
                "factor selector scope must contain distinct indices".into(),
            ));
        }
        if allowed.windows(2).any(|pair| pair[0] >= pair[1])
            || allowed
                .iter()
                .any(|pattern| pattern.bits() > scope.len() as u64)
        {
            return Err(PositivePacketContextError(
                "allowed factor patterns must be sorted, unique, and fit the scope".into(),
            ));
        }
        Ok(Self { scope, allowed })
    }

    pub fn scope(&self) -> &[u32] {
        &self.scope
    }

    pub fn allowed(&self) -> &[BigUint] {
        &self.allowed
    }
}

/// Fixed physical order of optional CNOT selectors and their local factors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionalGatePacket {
    n: u32,
    gates: Vec<(u32, u32)>,
    factors: Vec<LocalBooleanFactor>,
}

impl OptionalGatePacket {
    pub fn new(
        n: u32,
        gates: Vec<(u32, u32)>,
        factors: Vec<LocalBooleanFactor>,
    ) -> Result<Self, PositivePacketContextError> {
        if n == 0 {
            return Err(PositivePacketContextError(
                "positive physical width required".into(),
            ));
        }
        if gates
            .iter()
            .any(|(control, target)| *control >= n || *target >= n || control == target)
        {
            return Err(PositivePacketContextError(
                "each selector must name a valid physical CNOT".into(),
            ));
        }
        let slots = u32::try_from(gates.len()).map_err(|_| {
            PositivePacketContextError("selector count exceeds the index representation".into())
        })?;
        if factors
            .iter()
            .flat_map(|factor| &factor.scope)
            .any(|selector| *selector >= slots)
        {
            return Err(PositivePacketContextError(
                "factor selector index is outside the ordered gate packet".into(),
            ));
        }
        Ok(Self { n, gates, factors })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn gates(&self) -> &[(u32, u32)] {
        &self.gates
    }

    pub fn factors(&self) -> &[LocalBooleanFactor] {
        &self.factors
    }
}

mod compute;
mod producer;

pub use producer::{register_default_positive_packet_contexts, register_positive_packet_contexts};
