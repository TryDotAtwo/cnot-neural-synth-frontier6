use super::BellmanError;
use crate::packet::FeaturePacket;
use sha2::{Digest, Sha256};
use tch::nn::VarStore;
use tch::Device;

pub(super) fn theta_fingerprint(variables: &VarStore) -> Result<[u8; 32], BellmanError> {
    let mut named: Vec<_> = variables.variables().into_iter().collect();
    named.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    for (name, value) in named {
        hash_bytes(&mut digest, name.as_bytes());
        hash_bytes(&mut digest, format!("{:?}", value.kind()).as_bytes());
        hash_u64(&mut digest, value.size().len() as u64);
        for extent in value.size() {
            digest.update(extent.to_le_bytes());
        }
        let value = value.detach().to_device(Device::Cpu).contiguous();
        let byte_count = value
            .numel()
            .checked_mul(value.kind().elt_size_in_bytes())
            .ok_or_else(|| BellmanError::Invalid("parameter byte size overflow".into()))?;
        let mut bytes = vec![0u8; byte_count];
        value.f_copy_data_u8(&mut bytes, value.numel())?;
        hash_bytes(&mut digest, &bytes);
    }
    Ok(digest.finalize().into())
}

pub(super) fn packet_fingerprint(
    parent: &FeaturePacket,
    successor: Option<&FeaturePacket>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    hash_packet(&mut digest, parent);
    if let Some(successor) = successor {
        digest.update([1]);
        hash_packet(&mut digest, successor);
    } else {
        digest.update([0]);
    }
    digest.finalize().into()
}

fn hash_packet(digest: &mut Sha256, packet: &FeaturePacket) {
    hash_bytes(digest, packet.feature_id_manifest_id.as_bytes());
    hash_text(digest, &packet.schema_id);
    hash_text(digest, &packet.policy_id);
    hash_u64(digest, packet.batch as u64);
    hash_u64(digest, packet.nodes as u64);
    for value in &packet.valid_nodes {
        digest.update([*value]);
    }
    for (identity, alphabet) in packet.state_keys.iter().zip(&packet.alphabet_ids) {
        hash_u64(digest, u64::from(identity.dimension));
        hash_text(digest, &identity.alphabet_id);
        hash_text(digest, alphabet);
        hash_u64(digest, identity.packed_rows.len() as u64);
        for word in &identity.packed_rows {
            digest.update(word.to_le_bytes());
        }
    }
    hash_u64(digest, packet.entries.len() as u64);
    for entry in &packet.entries {
        hash_text(digest, &entry.spec.id.to_string());
        hash_text(digest, &format!("{:?}", entry.spec.role));
        hash_u64(digest, entry.spec.channels as u64);
        hash_text(digest, &entry.spec.scope);
        hash_text(digest, &entry.spec.producer_version);
        hash_u64(digest, entry.spec.dependencies.len() as u64);
        for dependency in &entry.spec.dependencies {
            hash_text(digest, dependency);
        }
        hash_text(digest, &entry.spec.exactness);
        hash_text(digest, &entry.spec.cost_tier);
        hash_u64(digest, entry.spec.wire_roles.len() as u64);
        for role in &entry.spec.wire_roles {
            hash_text(digest, role);
        }
        digest.update([u8::from(entry.spec.requires_solver_assisted)]);
        hash_u64(digest, entry.shape.len() as u64);
        for extent in &entry.shape {
            hash_u64(digest, *extent as u64);
        }
        for value in &entry.values {
            digest.update(value.to_bits().to_le_bytes());
        }
        for value in &entry.available {
            digest.update([*value]);
        }
        for status in &entry.status {
            hash_text(digest, status.as_str());
        }
        for reason in &entry.reason {
            hash_text(digest, reason);
        }
        for role in &entry.wire_roles {
            hash_text(digest, role);
        }
        hash_optional_values(digest, entry.token_valid.as_deref());
        hash_optional_f32(digest, entry.incidence.as_deref());
    }
}

fn hash_optional_values(digest: &mut Sha256, values: Option<&[u8]>) {
    match values {
        Some(values) => {
            digest.update([1]);
            hash_u64(digest, values.len() as u64);
            digest.update(values);
        }
        None => digest.update([0]),
    }
}

fn hash_optional_f32(digest: &mut Sha256, values: Option<&[f32]>) {
    match values {
        Some(values) => {
            digest.update([1]);
            hash_u64(digest, values.len() as u64);
            for value in values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        None => digest.update([0]),
    }
}

fn hash_text(digest: &mut Sha256, value: &str) {
    hash_bytes(digest, value.as_bytes());
}

fn hash_bytes(digest: &mut Sha256, value: &[u8]) {
    hash_u64(digest, value.len() as u64);
    digest.update(value);
}

fn hash_u64(digest: &mut Sha256, value: u64) {
    digest.update(value.to_le_bytes());
}

pub(super) fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
