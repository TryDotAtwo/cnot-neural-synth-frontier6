//! Complete 102-input H07 diagnostic. All matrix transforms, physical producers,
//! independent value-head forwards and the ordered FP64 reduction run on CUDA.
//! The single export is the final report: predictions, moments and status bytes.
#[cfg(test)]
pub(crate) mod tests;
mod views;
pub(crate) use views::problem_scope;

use super::statistics::{finish_spread, normalize_digest, validate_probe_metadata, Spread};
use super::{H07Diagnostic, H07Error, H07InputCoverage, H07InputCoverageEntry, H07ProbePrediction};
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeaturePacket, DeviceFeatureStatus, FusionDistanceField};
use crate::schema::{all_model_input_specs, FEATURE_ID_MANIFEST_ID, MODEL_INPUT_COUNT};
use crate::{CnotAlphabet, FeatureId, FeatureStatus, MatrixState};
use sha2::{Digest, Sha256};
use tch::{Device, Kind, Tensor};

fn moments(values: &Tensor) -> Tensor {
    let n = values.size()[0];
    let options = (Kind::Double, values.device());
    let mut mean = Tensor::zeros([], options);
    let mut sum = mean.zeros_like();
    for i in 0..n {
        let x = values.get(i);
        let delta = &x - &mean;
        mean += &delta / Tensor::full([], (i + 1) as f64, options);
        sum += delta * (&x - &mean);
    }
    let variance = sum / Tensor::full([], n as f64, options);
    let range = values.max() - values.min();
    let channel = variance.to_kind(Kind::Float).to_kind(Kind::Double);
    Tensor::stack(&[mean, variance, range, channel], 0)
}

// Each listed producer has a unique source reason for this status on a
// validated invertible binary matrix. The diagonal bit always belongs to its
// row and column weight, so the column producer's underflow case is impossible.
fn inapplicable_reason(id: FeatureId) -> Option<&'static str> {
    match (id.family(), id.ordinal()) {
        ('C', 5) | ('E', 1..=9) => Some("no_nonzero_partial_shear_block"),
        ('G', 1..=4) => Some("selected row/column class is outside the diffusion chain"),
        ('G', 6) => Some("matrix column pair is outside the rank-2 diffusion chain"),
        ('I', 7) => Some("i07_state_outside_explicit_stage52_interface"),
        _ => None,
    }
}

fn coverage(
    probe: &str,
    schema: &str,
    policy: &str,
    codes: &[f64],
) -> Result<H07InputCoverage, H07Error> {
    let error = |reason: String| H07Error::ModelInputCoverage {
        probe_id: probe.into(),
        reason,
    };
    if codes.len() != MODEL_INPUT_COUNT {
        return Err(error("requires exactly 102 model inputs".into()));
    }
    let mut entries = Vec::with_capacity(MODEL_INPUT_COUNT);
    for (spec, &code) in all_model_input_specs().into_iter().zip(codes) {
        let (status, available, reason) = match code as u8 {
            c if c == DeviceFeatureStatus::Computed as u8 => (FeatureStatus::Computed, 1, ""),
            c if c == DeviceFeatureStatus::Inapplicable as u8 => (
                FeatureStatus::Inapplicable,
                0,
                inapplicable_reason(spec.id).ok_or_else(|| {
                    error(format!("{} has no source inapplicability reason", spec.id))
                })?,
            ),
            3 if schema == crate::schema::MISSING_TOKEN_LAYOUT_ID => {
                (FeatureStatus::Unsupported, 0, "unsupported")
            }
            4 if schema == crate::schema::MISSING_TOKEN_LAYOUT_ID => {
                (FeatureStatus::ResourceLimit, 0, "resource_limit")
            }
            c => {
                let label = match c {
                    0 => "not_requested",
                    3 => "unsupported",
                    4 => "resource_limit",
                    _ => "invalid_status",
                };
                return Err(error(format!("{} {label}", spec.id)));
            }
        };
        entries.push(H07InputCoverageEntry {
            id: spec.id,
            status,
            available,
            reason: reason.into(),
        });
    }
    let result = H07InputCoverage {
        schema_id: schema.into(),
        policy_id: policy.into(),
        feature_id_manifest_id: FEATURE_ID_MANIFEST_ID.into(),
        entries,
    };
    if !result.is_complete() {
        return Err(error("incomplete canonical model-input coverage".into()));
    }
    Ok(result)
}

pub(crate) fn evaluate(
    root: &MatrixState,
    alphabet: &CnotAlphabet,
    model: &FusionDistanceField,
    plan: &DeviceFeaturePlan,
    model_contract: &str,
) -> Result<(serde_json::Value, H07Diagnostic), String> {
    let model_hash =
        normalize_digest("model contract digest", model_contract).map_err(|e| e.to_string())?;
    let scope = problem_scope(root, alphabet)?;
    let scope_hash = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&scope).map_err(|e| e.to_string())?)
    );
    let Device::Cuda(index) = model.device() else {
        return Err("resident H07 requires CUDA".into());
    };
    let _guard = crate::cuda::DeviceGuard::enter(index)?;
    let _no_grad = tch::no_grad_guard();
    let mut predictions = Vec::with_capacity(views::PROBES.len());
    let mut status = Vec::with_capacity(views::PROBES.len());
    let mut policies = Vec::with_capacity(views::PROBES.len());
    for ((id, _), packed) in views::PROBES
        .iter()
        .zip(views::packed_views(root, model.device()))
    {
        let state = DeviceStateBatch::from_packed_tensor_for_alphabet(packed, alphabet, index)?;
        let packet: DeviceFeaturePacket = plan.build(&state)?;
        status.push(
            packet
                .selected_status_matrix()
                .map_err(|e| e.to_string())?
                .view([-1]),
        );
        policies.push((packet.schema_id().to_owned(), packet.policy_id().to_owned()));
        let value = model
            .value_device(&packet)
            .map_err(|e| format!("{id}: {e}"))?;
        if value.numel() != 1 {
            return Err(H07Error::ModelOutputCardinality {
                probe_id: (*id).into(),
                outputs: value.numel(),
            }
            .to_string());
        }
        predictions.push(value.reshape([]).to_kind(Kind::Double));
    }
    let values = Tensor::stack(&predictions, 0);
    let exported = Tensor::cat(
        &[
            &values,
            &moments(&values),
            &Tensor::cat(&status, 0).to_kind(Kind::Double),
        ],
        0,
    )
    .to_device(Device::Cpu);
    let data = Vec::<f64>::try_from(&exported).map_err(|e| e.to_string())?;
    let count = views::PROBES.len();
    let mut probes = Vec::with_capacity(count);
    for (i, ((id, kind), (schema, policy))) in views::PROBES.iter().zip(policies).enumerate() {
        let start = count + 4 + i * MODEL_INPUT_COUNT;
        let audit = coverage(
            id,
            &schema,
            &policy,
            &data[start..start + MODEL_INPUT_COUNT],
        )
        .map_err(|e| e.to_string())?;
        probes.push(H07ProbePrediction {
            probe_id: (*id).into(),
            kind: *kind,
            prediction: data[i],
            input_coverage: Some(audit),
        });
    }
    let counts = validate_probe_metadata(&probes).map_err(|e| e.to_string())?;
    let report = finish_spread(
        model_hash,
        scope_hash,
        probes,
        counts,
        Spread::from_export(
            data[count..count + 4]
                .try_into()
                .expect("four device statistics"),
        ),
    )
    .map_err(|e| e.to_string())?;
    Ok((scope, report))
}

pub(crate) fn report_json(result: &H07Diagnostic) -> serde_json::Value {
    use serde_json::json;
    let mut probes = Vec::with_capacity(result.sample_count);
    for probe in &result.probes {
        let coverage = probe
            .input_coverage
            .as_ref()
            .expect("resident forwards always record coverage");
        let entries: Vec<_> = coverage
            .entries
            .iter()
            .map(|entry| {
                json!({
                    "id": entry.id.to_string(), "status": entry.status.as_str(),
                    "available": entry.available, "reason": entry.reason
                })
            })
            .collect();
        probes.push(json!({
            "probe_id": probe.probe_id, "kind": probe.kind.label(), "prediction": probe.prediction,
            "input_coverage": {
                "schema_id": coverage.schema_id, "policy_id": coverage.policy_id,
                "feature_id_manifest_id": coverage.feature_id_manifest_id, "complete": coverage.is_complete(),
                "entries": entries
            }
        }));
    }
    json!({
        "id": result.id.to_string(), "producer_version": result.producer_version,
        "model_contract_sha256": result.model_contract_sha256, "problem_scope_sha256": result.problem_scope_sha256,
        "probe_set_sha256": result.probe_set_sha256, "channel": result.channel, "mean": result.mean,
        "population_variance": result.population_variance, "range": result.range, "sample_count": result.sample_count,
        "view_sample_count": result.view_sample_count, "relabel_sample_count": result.relabel_sample_count, "probes": probes
    })
}
