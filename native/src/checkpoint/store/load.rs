use super::super::codec::{decode_checkpoint, validate_manifest_name};
use super::super::integrity::{canonical_bytes, canonical_value, digest, digest_reader};
use super::super::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
use serde_json::Value;
use std::fs::File;
use std::io::{Seek, SeekFrom};

impl CheckpointStore {
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
}
