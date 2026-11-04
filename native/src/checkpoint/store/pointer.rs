use super::super::codec::validate_manifest_name;
use super::super::integrity::canonical_bytes;
use super::super::{CheckpointError, CheckpointManifest, CheckpointStore};
use super::io::temporary_path;
use std::fs::{self, OpenOptions};
use std::io::Write;

impl CheckpointStore {
    pub(super) fn read_pointer(
        &self,
        name: &str,
    ) -> Result<Option<CheckpointManifest>, CheckpointError> {
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

    pub(super) fn write_pointer(
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
