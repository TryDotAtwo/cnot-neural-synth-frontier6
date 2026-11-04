use super::super::TchTrainingError;
use super::{AdamW, AdamWConfig, DecodedAdamW};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use tch::{Device, Tensor};

const OPTIMIZER_MAGIC: &[u8; 8] = b"C103ADW1";
const OPTIMIZER_VERSION: u32 = 1;
const MAX_OPTIMIZER_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterSignature {
    name: String,
    shape: Vec<i64>,
    kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdamWHeader {
    version: u32,
    config: AdamWConfig,
    steps: u64,
    parameter_steps: Vec<u64>,
    parameters: Vec<ParameterSignature>,
}

fn signatures(optimizer: &AdamW) -> Vec<ParameterSignature> {
    optimizer
        .parameters
        .iter()
        .map(|state| ParameterSignature {
            name: state.name.clone(),
            shape: state.parameter.size(),
            kind: format!("{:?}", state.parameter.kind()),
        })
        .collect()
}

pub(super) fn encode(optimizer: &AdamW) -> Result<Vec<u8>, TchTrainingError> {
    let header = AdamWHeader {
        version: OPTIMIZER_VERSION,
        config: optimizer.config,
        steps: optimizer.steps,
        parameter_steps: optimizer
            .parameters
            .iter()
            .map(|state| state.steps)
            .collect(),
        parameters: signatures(optimizer),
    };
    let header_bytes = serde_json::to_vec(&header)?;
    if header_bytes.len() > MAX_OPTIMIZER_HEADER_BYTES {
        return Err(TchTrainingError::Invalid(
            "AdamW metadata header exceeds size limit",
        ));
    }

    let mut named_tensors = Vec::with_capacity(optimizer.parameters.len() * 2);
    for (index, state) in optimizer.parameters.iter().enumerate() {
        named_tensors.push((
            format!("m_{index:08}"),
            state.first_moment.detach().to_device(Device::Cpu),
        ));
        named_tensors.push((
            format!("v_{index:08}"),
            state.second_moment.detach().to_device(Device::Cpu),
        ));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(OPTIMIZER_MAGIC);
    bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    Tensor::save_multi_to_stream(&named_tensors, &mut bytes)?;
    Ok(bytes)
}

pub(super) fn decode(optimizer: &AdamW, bytes: &[u8]) -> Result<DecodedAdamW, TchTrainingError> {
    if bytes.len() < OPTIMIZER_MAGIC.len() + 8 || &bytes[..8] != OPTIMIZER_MAGIC {
        return Err(TchTrainingError::Invalid("AdamW state magic differs"));
    }
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if header_len > MAX_OPTIMIZER_HEADER_BYTES {
        return Err(TchTrainingError::Invalid(
            "AdamW metadata header exceeds size limit",
        ));
    }
    let payload_start = 16usize
        .checked_add(header_len)
        .filter(|end| *end <= bytes.len())
        .ok_or(TchTrainingError::Invalid("AdamW state is truncated"))?;
    let header: AdamWHeader = serde_json::from_slice(&bytes[16..payload_start])?;
    if header.version != OPTIMIZER_VERSION
        || header.config != optimizer.config
        || header.parameters != signatures(optimizer)
        || header.parameter_steps.len() != optimizer.parameters.len()
        || header
            .parameter_steps
            .iter()
            .any(|steps| *steps > header.steps)
    {
        return Err(TchTrainingError::Invalid(
            "AdamW state is incompatible with this model or configuration",
        ));
    }
    let loaded = Tensor::load_multi_from_stream_with_device(
        Cursor::new(&bytes[payload_start..]),
        optimizer.parameters[0].parameter.device(),
    )?;
    let tensor_map: HashMap<_, _> = loaded.into_iter().collect();
    if tensor_map.len() != optimizer.parameters.len() * 2 {
        return Err(TchTrainingError::Invalid(
            "AdamW state tensor count differs",
        ));
    }

    let mut moments = Vec::with_capacity(optimizer.parameters.len());
    for (index, state) in optimizer.parameters.iter().enumerate() {
        let first = tensor_map
            .get(&format!("m_{index:08}"))
            .ok_or(TchTrainingError::Invalid("AdamW first moment is missing"))?;
        let second = tensor_map
            .get(&format!("v_{index:08}"))
            .ok_or(TchTrainingError::Invalid("AdamW second moment is missing"))?;
        if first.size() != state.parameter.size()
            || second.size() != state.parameter.size()
            || first.kind() != state.parameter.kind()
            || second.kind() != state.parameter.kind()
            || first.isfinite().all().int64_value(&[]) == 0
            || second.isfinite().all().int64_value(&[]) == 0
        {
            return Err(TchTrainingError::Invalid(
                "AdamW state tensor shape, type, or value differs",
            ));
        }
        moments.push((first.shallow_clone(), second.shallow_clone()));
    }
    Ok(DecodedAdamW {
        steps: header.steps,
        parameter_steps: header.parameter_steps,
        moments,
    })
}
