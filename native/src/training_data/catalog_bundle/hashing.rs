//! Streaming digest verification for catalog bundle bytes.

use sha2::{Digest, Sha256};

use super::super::TrainingDataError;

pub(super) struct CatalogHashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R> CatalogHashingReader<R> {
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    pub(super) fn finish(mut self) -> Result<String, TrainingDataError>
    where
        R: std::io::Read,
    {
        let mut buffer = [0u8; 1 << 16];
        loop {
            if std::io::Read::read(&mut self, &mut buffer).map_err(|error| {
                TrainingDataError(format!("finish hashing exact catalog: {error}"))
            })? == 0
            {
                break;
            }
        }
        Ok(self
            .hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

impl<R: std::io::Read> std::io::Read for CatalogHashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..count]);
        Ok(count)
    }
}
