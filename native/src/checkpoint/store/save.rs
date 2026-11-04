use super::super::codec::{
    validate_checkpoint, FileHeader, FORMAT_VERSION, MAGIC, MAX_HEADER_BYTES,
};
use super::super::integrity::{canonical_bytes, canonical_value, digest, digest_parts};
use super::super::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
use super::io::write_immutable;
use std::fs;

impl CheckpointStore {
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
}
