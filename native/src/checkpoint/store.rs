use super::codec::{
    decode_checkpoint, validate_checkpoint, validate_manifest_name, FileHeader, FORMAT_VERSION,
    MAGIC, MAX_HEADER_BYTES,
};
use super::integrity::{canonical_bytes, canonical_value, digest, digest_parts, digest_reader};
use super::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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
                if previous.sha256 != manifest.sha256 {
                    return Err(CheckpointError::ConflictingProgress);
                }
                match self.load_manifest(&previous, &header.contract) {
                    Ok(_) => {
                        if update_best {
                            self.write_pointer("best.json", &previous)?;
                        }
                        return Ok(previous);
                    }
                    // The caller still has the complete in-memory state, so a
                    // missing immutable file can be reconstructed below. A
                    // present but invalid file must remain a hard failure.
                    Err(CheckpointError::Io(error))
                        if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
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
