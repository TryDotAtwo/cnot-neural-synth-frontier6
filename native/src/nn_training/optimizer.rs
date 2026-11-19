mod codec;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
pub(super) use device::FlatAdamW;
mod config;
mod state;
mod update;

pub use config::AdamWConfig;
pub(super) use state::{AdamW, DecodedAdamW};
pub(super) const NONFINITE_CANDIDATE: &str = "AdamW candidate state contains non-finite values";
