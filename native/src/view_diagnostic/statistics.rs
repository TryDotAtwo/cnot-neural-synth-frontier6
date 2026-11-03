use super::types::{H07Diagnostic, H07Error, H07ProbeKind, H07ProbePrediction};
use super::H07_PRODUCER_VERSION;
use crate::schema::FeatureId;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

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

pub(super) fn validate_probe_metadata(
    probes: &[H07ProbePrediction],
) -> Result<(usize, usize), H07Error> {
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

pub(super) fn normalize_digest(name: &str, digest: &str) -> Result<String, H07Error> {
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
        match &probe.input_coverage {
            Some(coverage) => {
                hash_field(&mut digest, &[1]);
                hash_field(&mut digest, coverage.schema_id.as_bytes());
                hash_field(&mut digest, coverage.policy_id.as_bytes());
                hash_field(&mut digest, coverage.feature_id_manifest_id.as_bytes());
                hash_field(&mut digest, &(coverage.entries.len() as u64).to_le_bytes());
                for entry in &coverage.entries {
                    hash_field(&mut digest, entry.id.to_string().as_bytes());
                    hash_field(&mut digest, entry.status.as_str().as_bytes());
                    hash_field(&mut digest, &[entry.available]);
                    hash_field(&mut digest, entry.reason.as_bytes());
                }
            }
            None => hash_field(&mut digest, &[0]),
        }
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
