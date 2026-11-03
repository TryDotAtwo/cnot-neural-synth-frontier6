//! Public facade for sealed frozen-data loading and validation.
mod alphabet;
mod loader;
mod reader;
mod symmetry;
mod types;
mod wire;

pub use types::{FrozenDataError, FrozenDataset, FrozenPanel, FrozenRecordMetadata};

#[cfg(test)]
#[path = "frozen_data/tests.rs"]
mod tests;
