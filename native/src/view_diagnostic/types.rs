use crate::packet::FeatureStatus;
use crate::schema::{all_model_input_specs, FeatureId, FEATURE_ID_MANIFEST_ID, MODEL_INPUT_COUNT};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H07ProbeKind {
    Canonical,
    Transpose,
    Inverse,
    InverseTranspose,
    Relabel,
}

impl H07ProbeKind {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Transpose => "transpose",
            Self::Inverse => "inverse",
            Self::InverseTranspose => "inverse_transpose",
            Self::Relabel => "relabel",
        }
    }

    pub(super) fn is_equivalent_view(self) -> bool {
        matches!(
            self,
            Self::Canonical | Self::Transpose | Self::Inverse | Self::InverseTranspose
        )
    }
}

/// Per-input availability recorded for one actual H07 model forward.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H07InputCoverageEntry {
    pub id: FeatureId,
    pub status: FeatureStatus,
    pub available: u8,
    pub reason: String,
}

/// Exact packet policy and per-feature outcomes used by one H07 prediction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H07InputCoverage {
    pub schema_id: String,
    pub policy_id: String,
    pub feature_id_manifest_id: String,
    /// Canonical order, containing one record for each model input.
    pub entries: Vec<H07InputCoverageEntry>,
}

impl H07InputCoverage {
    pub fn selected_model_input_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status != FeatureStatus::NotRequested)
            .count()
    }

    pub fn computed_model_input_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == FeatureStatus::Computed && entry.available == 1)
            .count()
    }

    pub fn inapplicable_model_input_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == FeatureStatus::Inapplicable && entry.available == 0)
            .count()
    }

    /// Counts inputs that lack a computed or explicitly inapplicable result.
    pub fn unavailable_model_input_count(&self) -> usize {
        self.entries.len().saturating_sub(
            self.computed_model_input_count() + self.inapplicable_model_input_count(),
        )
    }

    /// True only for the exact canonical 102-input set with explicit outcomes.
    pub fn is_complete(&self) -> bool {
        let expected = all_model_input_specs();
        !self.schema_id.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && self.feature_id_manifest_id == FEATURE_ID_MANIFEST_ID
            && expected.len() == MODEL_INPUT_COUNT
            && self.entries.len() == MODEL_INPUT_COUNT
            && expected.iter().zip(&self.entries).all(|(spec, entry)| {
                spec.id == entry.id
                    && match entry.status {
                        FeatureStatus::Computed => entry.available == 1,
                        FeatureStatus::Inapplicable => entry.available == 0,
                        FeatureStatus::Unsupported
                        | FeatureStatus::ResourceLimit
                        | FeatureStatus::NotRequested => false,
                    }
            })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct H07ProbePrediction {
    /// Stable identifier for the exact transformed input/probe.
    pub probe_id: String,
    pub kind: H07ProbeKind,
    /// Scalar prediction for the same task scope in the canonical coordinates.
    pub prediction: f64,
    /// Present for runtime forwards; synthetic prediction-only computations may omit it.
    pub input_coverage: Option<H07InputCoverage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct H07Diagnostic {
    pub id: FeatureId,
    pub producer_version: String,
    pub model_contract_sha256: String,
    pub problem_scope_sha256: String,
    pub probe_set_sha256: String,
    /// Float32 scalar suitable for an external diagnostic channel. It is not a
    /// model input feature.
    pub channel: f32,
    pub mean: f64,
    pub population_variance: f64,
    pub range: f64,
    pub sample_count: usize,
    pub view_sample_count: usize,
    pub relabel_sample_count: usize,
    /// Canonically sorted samples retained for audit and reproduction.
    pub probes: Vec<H07ProbePrediction>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H07Error {
    InvalidDigest(String),
    InvalidProbeId,
    DuplicateProbeId(String),
    CanonicalProbeCount(usize),
    MissingEquivalentViewProbe,
    MissingRelabelProbe,
    NonFinitePrediction(String),
    VarianceNotRepresentable,
    ModelForward(String),
    ModelProbeBatch { probe_id: String, batch: usize },
    ModelOutputCardinality { probe_id: String, outputs: usize },
    ModelInputCoverage { probe_id: String, reason: String },
}

impl Display for H07Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDigest(name) => write!(formatter, "{name} must be 64 hexadecimal digits"),
            Self::InvalidProbeId => formatter.write_str("probe IDs must be nonempty and trimmed"),
            Self::DuplicateProbeId(id) => write!(formatter, "duplicate probe ID {id}"),
            Self::CanonicalProbeCount(count) => {
                write!(
                    formatter,
                    "exactly one canonical prediction is required, got {count}"
                )
            }
            Self::MissingEquivalentViewProbe => {
                formatter.write_str("at least one noncanonical equivalent-view probe is required")
            }
            Self::MissingRelabelProbe => {
                formatter.write_str("at least one wire-relabel probe is required")
            }
            Self::NonFinitePrediction(id) => {
                write!(formatter, "probe {id} has a non-finite prediction")
            }
            Self::VarianceNotRepresentable => {
                formatter.write_str("population variance cannot be represented as finite float32")
            }
            Self::ModelForward(error) => write!(formatter, "model forward failed: {error}"),
            Self::ModelProbeBatch { probe_id, batch } => write!(
                formatter,
                "probe {probe_id} must contain exactly one task, got batch {batch}"
            ),
            Self::ModelOutputCardinality { probe_id, outputs } => write!(
                formatter,
                "probe {probe_id} produced {outputs} scalar values; exactly one is required"
            ),
            Self::ModelInputCoverage { probe_id, reason } => {
                write!(
                    formatter,
                    "probe {probe_id} input coverage is malformed: {reason}"
                )
            }
        }
    }
}

impl Error for H07Error {}
