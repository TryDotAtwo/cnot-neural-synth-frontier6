use super::FeaturePolicy;
use crate::packet::FeatureStatus;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;

#[derive(Clone, Debug)]
pub struct FeatureValue {
    /// Shape excluding the batch axis.
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    /// Group-only metadata. `incidence` is `[groups, nodes, wire_roles]`.
    pub token_valid: Vec<u8>,
    pub incidence: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum FeatureComputation {
    Computed(FeatureValue),
    /// A computed tensor plus non-numeric per-row provenance that is retained
    /// in the packet sidecar and never appended to model inputs.
    ComputedWithProvenance {
        value: FeatureValue,
        provenance: Vec<String>,
    },
    Unavailable {
        status: FeatureStatus,
        reason: String,
    },
}

pub trait FeatureProducer: Send + Sync {
    fn spec(&self) -> &FeatureSpec;

    /// Names the actual symmetry contract implemented by the provider.
    fn symmetry_contract(&self) -> &str;

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation;
}
