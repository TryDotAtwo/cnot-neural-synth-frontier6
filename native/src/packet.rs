mod construction;
mod entry;
mod identity;
mod slicing;

pub use entry::{FeatureEntry, FeaturePacketError, FeatureStatus};
pub use identity::StateIdentity;

#[derive(Clone, Debug)]
pub struct FeaturePacket {
    pub entries: Vec<FeatureEntry>,
    pub state_keys: Vec<StateIdentity>,
    pub alphabet_ids: Vec<String>,
    pub batch: usize,
    pub nodes: usize,
    pub valid_nodes: Vec<u8>,
    pub schema_id: String,
    pub policy_id: String,
    pub feature_id_manifest_id: &'static str,
}
