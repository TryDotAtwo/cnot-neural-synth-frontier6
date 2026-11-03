//! Feature provider contracts, policy identity, and registry exports.
//!
//! Provider code is deliberately absent until an exact Rust implementation is
//! ported and reviewed. Missing providers are recorded as unavailable; they are
//! never replaced by zero-valued features marked as computed.

use crate::packet::{FeaturePacketError, FeatureStatus};
use crate::schema::{all_feature_specs, all_model_input_specs, FeatureId, FeatureSpec};
use crate::state::MatrixState;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureBuildError(pub String);

impl fmt::Display for FeatureBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FeatureBuildError {}

impl From<FeaturePacketError> for FeatureBuildError {
    fn from(error: FeaturePacketError) -> Self {
        Self(error.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePolicy {
    pub enabled_ids: Vec<FeatureId>,
    pub seed: u64,
    pub schema_id: String,
    pub solver_assisted: bool,
    pub symmetry_contract: String,
    pub evaluations: Option<usize>,
    pub max_bytes: Option<usize>,
}

impl FeaturePolicy {
    pub fn complete_103(schema_id: impl Into<String>) -> Self {
        Self {
            enabled_ids: all_feature_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect(),
            seed: 0,
            schema_id: schema_id.into(),
            solver_assisted: false,
            symmetry_contract: "exact_relabel".into(),
            evaluations: None,
            max_bytes: None,
        }
    }

    /// Selects the canonical 103-ID manifest with the weaker symmetry
    /// contract needed by seeded Monte Carlo and physical-wire probes.
    pub fn complete_103_augmentation_only(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::complete_103(schema_id);
        policy.symmetry_contract = "augmentation_only".into();
        policy
    }

    /// Select the executable model inputs from the canonical 103-ID manifest.
    /// H07 remains a post-forward diagnostic, not a model input.
    pub fn all_model_inputs(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::complete_103(schema_id);
        policy.enabled_ids = all_model_input_specs()
            .into_iter()
            .map(|spec| spec.id)
            .collect();
        policy
    }

    /// All model inputs with the weaker symmetry contract needed by seeded
    /// Monte Carlo and physical-wire probes.
    pub fn all_model_inputs_augmentation_only(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::all_model_inputs(schema_id);
        policy.symmetry_contract = "augmentation_only".into();
        policy
    }

    pub fn id(&self) -> Result<String, FeatureBuildError> {
        validate_policy(self)?;
        // Stable, length-delimited textual form suitable for logs and checkpoint binding.
        // The campaign layer adds a cryptographic digest when it commits artifacts.
        let mut body = String::new();
        push_field(&mut body, &self.schema_id);
        push_field(&mut body, &self.seed.to_string());
        push_field(
            &mut body,
            if self.solver_assisted {
                "solver=1"
            } else {
                "solver=0"
            },
        );
        push_field(&mut body, &self.symmetry_contract);
        push_field(
            &mut body,
            &self
                .evaluations
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        push_field(
            &mut body,
            &self
                .max_bytes
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        for id in &self.enabled_ids {
            push_field(&mut body, &id.to_string());
        }
        Ok(body)
    }
}

fn push_field(output: &mut String, value: &str) {
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}

fn validate_policy(policy: &FeaturePolicy) -> Result<(), FeatureBuildError> {
    if policy.enabled_ids.is_empty() {
        return Err(FeatureBuildError(
            "feature policy selects no feature IDs".into(),
        ));
    }
    if policy.schema_id.trim().is_empty() || policy.symmetry_contract.trim().is_empty() {
        return Err(FeatureBuildError(
            "feature policy needs schema and symmetry identities".into(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(policy.enabled_ids.len());
    if policy.enabled_ids.iter().any(|id| !seen.insert(*id)) {
        return Err(FeatureBuildError(
            "feature policy repeats a feature ID".into(),
        ));
    }
    Ok(())
}

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

/// Explicit physical/experiment contexts needed to register every current model-input provider.
///
/// No contexts are synthesized from matrix width. Each collection must contain
/// enough validated, source-bound records for its provider; registration fails
/// closed when a required collection or resource bound is absent.
pub struct FeatureProviderContexts {
    pub max_n: u32,
    pub labeled_cuts: Vec<crate::rank_features::LabeledCut>,
    pub restricted_intersections: Vec<crate::rank_features::RestrictedIntersection>,
    pub rank_increment_probes: Vec<crate::rank_features::RankIncrementProbe>,
    pub alphabet: crate::alphabet::CnotAlphabet,
    pub selected_residual_gates: Vec<crate::state::CnotGate>,
    pub block_parity_contexts: Vec<crate::block_parities::BlockContext>,
    pub max_parity_patterns: usize,
    pub max_parity_pairs: usize,
    pub matrix_boundaries: Vec<crate::matrix_boundary::MatrixBoundarySpec>,
    pub max_boundary_ports: usize,
    pub max_boundary_equations: usize,
    pub paid_basis_contexts: Vec<crate::paid_basis_features::PaidBasisBlockContext>,
    pub paid_shear_config: crate::paid_basis::PaidShearConfig,
    pub template_response: crate::template_response::TemplateResponseConfig,
    pub bilinear_contexts: Vec<crate::bilinear::BilinearContext>,
    pub shared_relation_contexts: Vec<crate::shared_relation::SharedRelationContext>,
    pub frequency_contexts: Vec<crate::frequency_context::FrequencyContext>,
    pub positive_packet_contexts: Vec<crate::positive_packet::OptionalGatePacket>,
    pub i09_contexts: Vec<crate::i09_quadratic::I09Context>,
    pub i09_limits: crate::i09_quadratic::I09Limits,
    pub i10_config: crate::response::I10Config,
}

mod packet_builder;
mod registry;
pub use registry::FeatureRegistry;
