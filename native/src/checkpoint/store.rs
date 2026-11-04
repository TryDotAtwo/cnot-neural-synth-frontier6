mod io;
mod load;
mod pointer;
mod save;

use super::CheckpointStore;
use std::path::{Path, PathBuf};

impl CheckpointStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}
