use super::super::integrity::digest_reader;
use super::super::CheckpointError;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn temporary_path(root: &Path, name: &str) -> PathBuf {
    let suffix = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    root.join(format!(".{name}.{}.{}.tmp", std::process::id(), suffix))
}

pub(super) fn sync_directory(path: &Path) -> Result<(), CheckpointError> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_immutable(
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
            sync_directory(path.parent().ok_or(CheckpointError::Invalid(
                "checkpoint path has no parent",
            ))?)?;
            return Ok(());
        }
        return Err(CheckpointError::HashMismatch);
    }
    let parent = path
        .parent()
        .ok_or(CheckpointError::Invalid("checkpoint path has no parent"))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(CheckpointError::Invalid("checkpoint filename is not UTF-8"))?;
    let temporary = temporary_path(parent, name);
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
            Ok(()) => {
                sync_directory(parent)?;
                Ok(())
            },
            Err(error) if path.exists() => {
                let mut existing = File::open(path)?;
                if digest_reader(&mut existing)? == expected_sha256 {
                    sync_directory(parent)?;
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
