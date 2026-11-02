//! H07 post-forward prediction-spread diagnostic.
//!
//! The pinned feature schema keeps H07 unimplemented as a model input and gives
//! it a group/1 placeholder. The normative feature bank describes additional
//! forwards over equivalent views and relabel probes. This module runs those
//! forwards on caller-built packets and emits an auditable diagnostic; probe
//! construction and the exact source availability policy remain explicit.

use crate::schema::FeatureId;
#[cfg(feature = "torch")]
use crate::{model::FusionDistanceField, packet::FeaturePacket};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
#[cfg(feature = "torch")]
use tch::{Device, Kind};

pub const H07_PRODUCER_VERSION: &str = "h07_distance_field_value_population_variance_v2";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H07ProbeKind {
    Canonical,
    Transpose,
    Inverse,
    InverseTranspose,
    Relabel,
}

impl H07ProbeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Transpose => "transpose",
            Self::Inverse => "inverse",
            Self::InverseTranspose => "inverse_transpose",
            Self::Relabel => "relabel",
        }
    }

    fn is_equivalent_view(self) -> bool {
        matches!(
            self,
            Self::Canonical | Self::Transpose | Self::Inverse | Self::InverseTranspose
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct H07ProbePrediction {
    /// Stable identifier for the exact transformed input/probe.
    pub probe_id: String,
    pub kind: H07ProbeKind,
    /// Scalar prediction for the same task scope in the canonical coordinates.
    pub prediction: f64,
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
        }
    }
}

impl Error for H07Error {}

/// Computes H07 as a post-forward population variance over the canonical view,
/// equivalent matrix views, and wire-relabel probes.
///
/// `problem_scope_sha256` must bind the matrix task, allowed alphabet/costs, and
/// symmetry assumptions. The caller is responsible for generating only valid
/// cost-preserving probes and mapping every scalar back to the same task scope.
pub fn compute_h07_prediction_spread(
    model_contract_sha256: &str,
    problem_scope_sha256: &str,
    samples: &[H07ProbePrediction],
) -> Result<H07Diagnostic, H07Error> {
    let model_contract_sha256 = normalize_digest("model contract digest", model_contract_sha256)?;
    let problem_scope_sha256 = normalize_digest("problem scope digest", problem_scope_sha256)?;
    let mut probes = samples.to_vec();
    probes.sort_by(|left, right| left.probe_id.cmp(&right.probe_id));
    let (view_sample_count, relabel_sample_count) = validate_probe_metadata(&probes)?;

    let sample_count = probes.len();
    let mut mean = 0.0f64;
    let mut sum_squared_deviations = 0.0f64;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for (index, probe) in probes.iter().enumerate() {
        let value = probe.prediction;
        let count = (index + 1) as f64;
        let delta = value - mean;
        mean += delta / count;
        let delta_after = value - mean;
        sum_squared_deviations += delta * delta_after;
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let population_variance = sum_squared_deviations / sample_count as f64;
    let range = maximum - minimum;
    let channel = population_variance as f32;
    if !mean.is_finite()
        || !population_variance.is_finite()
        || !range.is_finite()
        || !channel.is_finite()
    {
        return Err(H07Error::VarianceNotRepresentable);
    }

    let probe_set_sha256 = probe_digest(&model_contract_sha256, &problem_scope_sha256, &probes);
    Ok(H07Diagnostic {
        id: FeatureId::new(b'H', 7).expect("H07 is in the canonical manifest"),
        producer_version: H07_PRODUCER_VERSION.into(),
        model_contract_sha256,
        problem_scope_sha256,
        probe_set_sha256,
        channel,
        mean,
        population_variance,
        range,
        sample_count,
        view_sample_count,
        relabel_sample_count,
        probes,
    })
}

fn validate_probe_metadata(probes: &[H07ProbePrediction]) -> Result<(usize, usize), H07Error> {
    let mut probe_ids = HashSet::with_capacity(probes.len());
    let mut canonical_count = 0usize;
    let mut view_sample_count = 0usize;
    let mut relabel_sample_count = 0usize;
    let mut has_noncanonical_view = false;
    for probe in probes {
        if probe.probe_id.trim().is_empty() || probe.probe_id.trim() != probe.probe_id {
            return Err(H07Error::InvalidProbeId);
        }
        if !probe_ids.insert(probe.probe_id.as_str()) {
            return Err(H07Error::DuplicateProbeId(probe.probe_id.clone()));
        }
        if !probe.prediction.is_finite() {
            return Err(H07Error::NonFinitePrediction(probe.probe_id.clone()));
        }
        if probe.kind == H07ProbeKind::Canonical {
            canonical_count += 1;
        }
        if probe.kind.is_equivalent_view() {
            view_sample_count += 1;
            has_noncanonical_view |= probe.kind != H07ProbeKind::Canonical;
        }
        if probe.kind == H07ProbeKind::Relabel {
            relabel_sample_count += 1;
        }
    }
    if canonical_count != 1 {
        return Err(H07Error::CanonicalProbeCount(canonical_count));
    }
    if !has_noncanonical_view {
        return Err(H07Error::MissingEquivalentViewProbe);
    }
    if relabel_sample_count == 0 {
        return Err(H07Error::MissingRelabelProbe);
    }
    Ok((view_sample_count, relabel_sample_count))
}

/// A single already-transformed model input for the H07 post-forward diagnostic.
///
/// The caller must construct only valid, cost-preserving view/relabel packets
/// and bind the exact task, alphabet/cost graph, and symmetry assumptions in
/// problem_scope_sha256.
#[cfg(feature = "torch")]
#[derive(Debug)]
pub struct H07ModelProbe<'a> {
    pub probe_id: String,
    pub kind: H07ProbeKind,
    pub packet: &'a FeaturePacket,
}

/// Runs one actual model forward per supplied canonical/view/relabel packet and
/// summarizes the scalar value-head predictions as H07. The source defines
/// H07 as an extra-forward diagnostic/ensemble, not as a model input channel;
/// probe generation remains explicit so invalid task symmetries cannot be
/// silently assumed.
/// The supplied model digest must bind the exact weights and model contract;
/// the problem-scope digest must bind the task, alphabet/cost graph, and probe
/// transformations used to construct the packets.
#[cfg(feature = "torch")]
pub fn evaluate_h07_prediction_spread(
    model: &FusionDistanceField,
    model_contract_sha256: &str,
    problem_scope_sha256: &str,
    probes: &[H07ModelProbe<'_>],
) -> Result<H07Diagnostic, H07Error> {
    let model_contract_sha256 = normalize_digest("model contract digest", model_contract_sha256)?;
    let problem_scope_sha256 = normalize_digest("problem scope digest", problem_scope_sha256)?;
    let metadata: Vec<_> = probes
        .iter()
        .map(|probe| H07ProbePrediction {
            probe_id: probe.probe_id.clone(),
            kind: probe.kind,
            prediction: 0.0,
        })
        .collect();
    validate_probe_metadata(&metadata)?;

    let mut predictions = Vec::with_capacity(probes.len());
    for probe in probes {
        if probe.packet.batch != 1 {
            return Err(H07Error::ModelProbeBatch {
                probe_id: probe.probe_id.clone(),
                batch: probe.packet.batch,
            });
        }
        let output = tch::no_grad(|| model.forward(probe.packet))
            .map_err(|error| H07Error::ModelForward(format!("{}: {error}", probe.probe_id)))?;
        let outputs = output.value.numel();
        if outputs != 1 {
            return Err(H07Error::ModelOutputCardinality {
                probe_id: probe.probe_id.clone(),
                outputs,
            });
        }
        let value = output
            .value
            .to_device(Device::Cpu)
            .to_kind(Kind::Double)
            .view([-1])
            .double_value(&[0]);
        predictions.push(H07ProbePrediction {
            probe_id: probe.probe_id.clone(),
            kind: probe.kind,
            prediction: value,
        });
    }
    compute_h07_prediction_spread(&model_contract_sha256, &problem_scope_sha256, &predictions)
}

fn normalize_digest(name: &str, digest: &str) -> Result<String, H07Error> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(H07Error::InvalidDigest(name.into()));
    }
    Ok(digest.to_ascii_lowercase())
}

fn probe_digest(
    model_contract_sha256: &str,
    problem_scope_sha256: &str,
    probes: &[H07ProbePrediction],
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, H07_PRODUCER_VERSION.as_bytes());
    hash_field(&mut digest, model_contract_sha256.as_bytes());
    hash_field(&mut digest, problem_scope_sha256.as_bytes());
    hash_field(&mut digest, &(probes.len() as u64).to_le_bytes());
    for probe in probes {
        hash_field(&mut digest, probe.probe_id.as_bytes());
        hash_field(&mut digest, probe.kind.label().as_bytes());
        hash_field(&mut digest, &probe.prediction.to_bits().to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}
