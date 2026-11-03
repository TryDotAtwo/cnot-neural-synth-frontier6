//! Streaming SHA-256 reader used to verify sealed artifact bytes.
use super::types::FrozenDataError;
use sha2::{Digest, Sha256};
use std::io::Read;

pub(super) struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}
impl<R> HashingReader<R> {
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }
    pub(super) fn finish(mut self) -> Result<String, FrozenDataError>
    where
        R: Read,
    {
        let mut buffer = [0u8; 1 << 16];
        loop {
            if self
                .read(&mut buffer)
                .map_err(|e| FrozenDataError(format!("finish hashing dataset: {e}")))?
                == 0
            {
                break;
            }
        }
        Ok(self
            .hasher
            .finalize()
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect())
    }
}
impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buf)?;
        self.hasher.update(&buf[..count]);
        Ok(count)
    }
}
