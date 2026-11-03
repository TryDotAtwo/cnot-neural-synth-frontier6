use super::integrity::{
    canonical_bytes, canonical_value, digest, digest_parts, is_sha256, read_blob,
};
use super::{Checkpoint, CheckpointError, CheckpointManifest};
use crate::training::{Progress, Sampler, SamplerSnapshot, TrainingError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path};

pub(super) const MAGIC: &[u8; 8] = b"C103QCP1";
pub(super) const FORMAT_VERSION: u32 = 1;
pub(super) const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileHeader {
    pub(super) version: u32,
    pub(super) contract: Value,
    pub(super) contract_sha256: String,
    pub(super) progress: Progress,
    pub(super) sampler: SamplerSnapshot,
    pub(super) model_len: u64,
    pub(super) optimizer_len: u64,
    pub(super) objective_len: u64,
    pub(super) rng_len: u64,
    pub(super) payload_sha256: String,
}

pub(super) fn validate_checkpoint(checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
    if !checkpoint.progress.validate() {
        return Err(CheckpointError::Training(TrainingError::InvalidProgress));
    }
    let mut sampler = Sampler::new(checkpoint.sampler.size, 0)?;
    sampler.restore(&checkpoint.sampler)?;
    canonical_bytes(&canonical_value(checkpoint.contract.clone()))?;
    Ok(())
}

pub(super) fn validate_manifest_name(manifest: &CheckpointManifest) -> Result<(), CheckpointError> {
    let path = Path::new(&manifest.file);
    if manifest.version != FORMAT_VERSION
        || path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
        || !manifest.file.ends_with(".qcp")
        || !is_sha256(&manifest.sha256)
        || !is_sha256(&manifest.contract_sha256)
        || !manifest.progress.validate()
    {
        return Err(CheckpointError::Invalid("malformed checkpoint pointer"));
    }
    Ok(())
}

pub(super) fn decode_checkpoint(
    file: &mut File,
    expected_contract: &Value,
) -> Result<Checkpoint, CheckpointError> {
    let file_len = file.metadata()?.len();
    if file_len < (MAGIC.len() + 8) as u64 {
        return Err(CheckpointError::Invalid(
            "magic or header length is missing",
        ));
    }
    let mut magic = [0u8; MAGIC.len()];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(CheckpointError::Invalid("checkpoint magic differs"));
    }
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| CheckpointError::Invalid("header length exceeds address space"))?;
    if header_len_usize > MAX_HEADER_BYTES {
        return Err(CheckpointError::Invalid(
            "metadata header exceeds size limit",
        ));
    }
    let mut header_bytes = Vec::new();
    header_bytes
        .try_reserve_exact(header_len_usize)
        .map_err(|_| CheckpointError::AllocationFailed("metadata"))?;
    header_bytes.resize(header_len_usize, 0);
    file.read_exact(&mut header_bytes)?;
    let header: FileHeader = serde_json::from_slice(&header_bytes)?;
    if header.version != FORMAT_VERSION || !header.progress.validate() {
        return Err(CheckpointError::Invalid(
            "unsupported version or progress counters",
        ));
    }
    let contract = canonical_value(header.contract);
    let contract_bytes = canonical_bytes(&contract)?;
    if digest(&contract_bytes) != header.contract_sha256 {
        return Err(CheckpointError::HashMismatch);
    }
    if contract != canonical_value(expected_contract.clone()) {
        return Err(CheckpointError::ContractMismatch);
    }
    let expected_file_len = (MAGIC.len() as u64 + 8)
        .checked_add(header_len)
        .and_then(|size| size.checked_add(header.model_len))
        .and_then(|size| size.checked_add(header.optimizer_len))
        .and_then(|size| size.checked_add(header.objective_len))
        .and_then(|size| size.checked_add(header.rng_len))
        .ok_or(CheckpointError::Invalid("payload lengths overflow"))?;
    if expected_file_len != file_len {
        return Err(CheckpointError::Invalid(
            "payload lengths do not match file size",
        ));
    }
    let model_state = read_blob(file, header.model_len, "model")?;
    let optimizer_state = read_blob(file, header.optimizer_len, "optimizer")?;
    let objective_state = read_blob(file, header.objective_len, "objective")?;
    let rng_state = read_blob(file, header.rng_len, "random state")?;
    if digest_parts(&[&model_state, &optimizer_state, &objective_state, &rng_state])
        != header.payload_sha256
    {
        return Err(CheckpointError::HashMismatch);
    }
    let checkpoint = Checkpoint {
        contract,
        progress: header.progress,
        sampler: header.sampler,
        model_state,
        optimizer_state,
        objective_state,
        rng_state,
    };
    validate_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}
