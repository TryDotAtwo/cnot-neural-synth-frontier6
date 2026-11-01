//! Immutable, hash-checked training snapshots with atomic `last`/`best` pointers.
//!
//! Model, optimizer, and objective adapters own their binary encodings. The
//! store binds those bytes to the feature/data/objective contract and to the
//! exact sampler and progress state, then verifies all bytes before returning
//! a checkpoint to a caller.

use crate::training::{Progress, Sampler, SamplerSnapshot, TrainingError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"C103QCP1";
const FORMAT_VERSION: u32 = 1;
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Training(TrainingError),
    Invalid(&'static str),
    AllocationFailed(&'static str),
    ContractMismatch,
    HashMismatch,
    ProgressRegression,
    ConflictingProgress,
}

impl Display for CheckpointError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "checkpoint I/O failed: {error}"),
            Self::Json(error) => write!(f, "checkpoint metadata is invalid: {error}"),
            Self::Training(error) => write!(f, "checkpoint training state is invalid: {error}"),
            Self::Invalid(reason) => write!(f, "invalid checkpoint: {reason}"),
            Self::AllocationFailed(which) => {
                write!(f, "could not reserve memory for checkpoint {which}")
            }
            Self::ContractMismatch => f.write_str("checkpoint contract differs from this run"),
            Self::HashMismatch => f.write_str("checkpoint hash does not match its contents"),
            Self::ProgressRegression => f.write_str("checkpoint progress cannot move backwards"),
            Self::ConflictingProgress => {
                f.write_str("different checkpoint contents already exist at this update")
            }
        }
    }
}

impl Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<TrainingError> for CheckpointError {
    fn from(error: TrainingError) -> Self {
        Self::Training(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub contract: Value,
    pub progress: Progress,
    pub sampler: SamplerSnapshot,
    pub model_state: Vec<u8>,
    pub optimizer_state: Vec<u8>,
    pub objective_state: Vec<u8>,
    pub rng_state: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    pub version: u32,
    pub file: String,
    pub sha256: String,
    pub contract_sha256: String,
    pub progress: Progress,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileHeader {
    version: u32,
    contract: Value,
    contract_sha256: String,
    progress: Progress,
    sampler: SamplerSnapshot,
    model_len: u64,
    optimizer_len: u64,
    objective_len: u64,
    rng_len: u64,
    payload_sha256: String,
}

/// Checkpoint persistence rooted in one training run directory.
#[derive(Debug, Clone)]
pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Write immutable state first; publish `last.json` only after the state
    /// bytes and metadata have both reached stable storage.
    pub fn save(
        &self,
        checkpoint: &Checkpoint,
        update_best: bool,
    ) -> Result<CheckpointManifest, CheckpointError> {
        validate_checkpoint(checkpoint)?;
        fs::create_dir_all(&self.root)?;

        let contract = canonical_value(checkpoint.contract.clone());
        let contract_sha256 = digest(&canonical_bytes(&contract)?);
        let payload_sha256 = digest_parts(&[
            &checkpoint.model_state,
            &checkpoint.optimizer_state,
            &checkpoint.objective_state,
            &checkpoint.rng_state,
        ]);
        let header = FileHeader {
            version: FORMAT_VERSION,
            contract,
            contract_sha256: contract_sha256.clone(),
            progress: checkpoint.progress,
            sampler: checkpoint.sampler.clone(),
            model_len: checkpoint.model_state.len() as u64,
            optimizer_len: checkpoint.optimizer_state.len() as u64,
            objective_len: checkpoint.objective_state.len() as u64,
            rng_len: checkpoint.rng_state.len() as u64,
            payload_sha256,
        };
        let header_bytes = canonical_bytes(&header)?;
        if header_bytes.len() > MAX_HEADER_BYTES {
            return Err(CheckpointError::Invalid(
                "metadata header exceeds size limit",
            ));
        }
        let header_len = (header_bytes.len() as u64).to_le_bytes();
        let file_sha256 = digest_parts(&[
            MAGIC,
            &header_len,
            &header_bytes,
            &checkpoint.model_state,
            &checkpoint.optimizer_state,
            &checkpoint.objective_state,
            &checkpoint.rng_state,
        ]);
        let file = format!(
            "step-{:012}-{}.qcp",
            checkpoint.progress.attempted, file_sha256
        );
        let manifest = CheckpointManifest {
            version: FORMAT_VERSION,
            file,
            sha256: file_sha256,
            contract_sha256,
            progress: checkpoint.progress,
        };

        if let Some(previous) = self.read_pointer("last.json")? {
            if previous.contract_sha256 != manifest.contract_sha256 {
                return Err(CheckpointError::ContractMismatch);
            }
            if previous.progress.attempted > manifest.progress.attempted {
                return Err(CheckpointError::ProgressRegression);
            }
            if previous.progress.attempted == manifest.progress.attempted {
                if previous.sha256 == manifest.sha256 {
                    return Ok(previous);
                }
                return Err(CheckpointError::ConflictingProgress);
            }
        }

        let file_path = self.root.join(&manifest.file);
        write_immutable(
            &file_path,
            MAGIC,
            &header_len,
            &header_bytes,
            &checkpoint.model_state,
            &checkpoint.optimizer_state,
            &checkpoint.objective_state,
            &checkpoint.rng_state,
            &manifest.sha256,
        )?;
        self.write_pointer("last.json", &manifest)?;
        if update_best {
            self.write_pointer("best.json", &manifest)?;
        }
        Ok(manifest)
    }

    pub fn load_latest(&self, expected_contract: &Value) -> Result<Checkpoint, CheckpointError> {
        let manifest = self
            .read_pointer("last.json")?
            .ok_or(CheckpointError::Invalid("last.json is missing"))?;
        self.load_manifest(&manifest, expected_contract)
    }

    pub fn load_best(&self, expected_contract: &Value) -> Result<Checkpoint, CheckpointError> {
        let manifest = self
            .read_pointer("best.json")?
            .ok_or(CheckpointError::Invalid("best.json is missing"))?;
        self.load_manifest(&manifest, expected_contract)
    }

    pub fn load_manifest(
        &self,
        manifest: &CheckpointManifest,
        expected_contract: &Value,
    ) -> Result<Checkpoint, CheckpointError> {
        validate_manifest_name(manifest)?;
        let path = self.root.join(&manifest.file);
        let mut file = File::open(path)?;
        if digest_reader(&mut file)? != manifest.sha256 {
            return Err(CheckpointError::HashMismatch);
        }
        file.seek(SeekFrom::Start(0))?;
        let checkpoint = decode_checkpoint(&mut file, expected_contract)?;
        if checkpoint.progress != manifest.progress
            || digest(&canonical_bytes(&canonical_value(
                checkpoint.contract.clone(),
            ))?) != manifest.contract_sha256
        {
            return Err(CheckpointError::Invalid(
                "pointer and snapshot metadata differ",
            ));
        }
        Ok(checkpoint)
    }

    fn read_pointer(&self, name: &str) -> Result<Option<CheckpointManifest>, CheckpointError> {
        let path = self.root.join(name);
        match fs::read(path) {
            Ok(bytes) => {
                let manifest: CheckpointManifest = serde_json::from_slice(&bytes)?;
                validate_manifest_name(&manifest)?;
                Ok(Some(manifest))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn write_pointer(
        &self,
        name: &str,
        manifest: &CheckpointManifest,
    ) -> Result<(), CheckpointError> {
        let path = self.root.join(name);
        let bytes = canonical_bytes(manifest)?;
        let temporary = temporary_path(&self.root, name);
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temporary, &path)?;
            Ok::<(), CheckpointError>(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
}

fn validate_checkpoint(checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
    if !checkpoint.progress.validate() {
        return Err(CheckpointError::Training(TrainingError::InvalidProgress));
    }
    let mut sampler = Sampler::new(checkpoint.sampler.size, 0)?;
    sampler.restore(&checkpoint.sampler)?;
    canonical_bytes(&canonical_value(checkpoint.contract.clone()))?;
    Ok(())
}

fn validate_manifest_name(manifest: &CheckpointManifest) -> Result<(), CheckpointError> {
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

fn decode_checkpoint(
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

fn canonical_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_value).collect()),
        Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut result = Map::new();
            for (key, value) in entries {
                result.insert(key, canonical_value(value));
            }
            Value::Object(result)
        }
        other => other,
    }
}

fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, CheckpointError> {
    Ok(serde_json::to_vec(value)?)
}

fn digest(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    to_hex(&hash)
}

fn digest_parts(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    to_hex(&hash.finalize())
}

fn digest_reader(reader: &mut File) -> Result<String, CheckpointError> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(to_hex(&hash.finalize()))
}

fn read_blob(
    file: &mut File,
    length: u64,
    label: &'static str,
) -> Result<Vec<u8>, CheckpointError> {
    let length = usize::try_from(length)
        .map_err(|_| CheckpointError::Invalid("payload length exceeds address space"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| CheckpointError::AllocationFailed(label))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn temporary_path(root: &Path, name: &str) -> PathBuf {
    let suffix = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    root.join(format!(".{name}.{}.{}.tmp", std::process::id(), suffix))
}

#[allow(clippy::too_many_arguments)]
fn write_immutable(
    path: &Path,
    magic: &[u8; 8],
    header_len: &[u8; 8],
    header: &[u8],
    model: &[u8],
    optimizer: &[u8],
    objective: &[u8],
    rng: &[u8],
    expected_sha256: &str,
) -> Result<(), CheckpointError> {
    if path.exists() {
        let mut existing = File::open(path)?;
        if digest_reader(&mut existing)? == expected_sha256 {
            return Ok(());
        }
        return Err(CheckpointError::HashMismatch);
    }
    let temporary = temporary_path(
        path.parent()
            .ok_or(CheckpointError::Invalid("checkpoint path has no parent"))?,
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or(CheckpointError::Invalid("checkpoint filename is not UTF-8"))?,
    );
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(magic)?;
        file.write_all(header_len)?;
        file.write_all(header)?;
        file.write_all(model)?;
        file.write_all(optimizer)?;
        file.write_all(objective)?;
        file.write_all(rng)?;
        file.sync_all()?;
        match fs::rename(&temporary, path) {
            Ok(()) => Ok(()),
            Err(error) if path.exists() => {
                let mut existing = File::open(path)?;
                if digest_reader(&mut existing)? == expected_sha256 {
                    let _ = fs::remove_file(&temporary);
                    Ok(())
                } else {
                    Err(error.into())
                }
            }
            Err(error) => Err(error.into()),
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
