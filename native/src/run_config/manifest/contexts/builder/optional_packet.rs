//! Conversion of optional-gate packet factors.

use super::super::OptionalGatePacketSpec;
use crate::{LocalBooleanFactor, OptionalGatePacket};
use num_bigint::BigUint;

impl OptionalGatePacketSpec {
    pub(crate) fn build(self) -> Result<OptionalGatePacket, String> {
        let gates = self.gates.into_iter().map(|[c, t]| (c, t)).collect();
        let factors = self
            .factors
            .into_iter()
            .map(|x| {
                let allowed = x
                    .allowed
                    .iter()
                    .map(|s| {
                        s.parse::<BigUint>()
                            .map_err(|e| format!("invalid decimal BigUint: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                LocalBooleanFactor::new(x.scope, allowed).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        OptionalGatePacket::new(self.n, gates, factors).map_err(|e| e.to_string())
    }
}
