use super::coverage::h07_input_coverage;
use super::statistics::{compute_h07_prediction_spread, normalize_digest, validate_probe_metadata};
use super::types::{H07Diagnostic, H07Error, H07ProbeKind, H07ProbePrediction};
use crate::model::FusionDistanceField;
use crate::packet::FeaturePacket;
use tch::{Device, Kind};

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
            input_coverage: None,
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
        let input_coverage = h07_input_coverage(probe.packet, &probe.probe_id)?;
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
            input_coverage: Some(input_coverage),
        });
    }
    compute_h07_prediction_spread(&model_contract_sha256, &problem_scope_sha256, &predictions)
}
