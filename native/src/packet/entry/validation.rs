//! Validation facade for dense feature-entry buffers.

mod schema;
mod storage;

use super::{FeatureEntry, FeaturePacketError};

impl FeatureEntry {
    pub fn validate(&self, batch: usize, nmax: usize) -> Result<(), FeaturePacketError> {
        schema::validate(self, batch, nmax)?;
        storage::validate(self, batch)?;
        Ok(())
    }
}
