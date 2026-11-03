use super::FeatureRegistry;
use crate::schema::{FeatureRole, FEATURE_ID_MANIFEST_ID, FEATURE_LAYOUT_ID};
use crate::structural;
use sha2::{Digest, Sha256};

impl FeatureRegistry {
    /// Stable digest of every feature layout/provider contract in canonical
    /// manifest order. Provider configuration belongs in `producer_version`.
    pub fn contract_sha256(&self) -> String {
        let mut digest = Sha256::new();
        hash_contract_field(&mut digest, b"cnot103_feature_registry_contract_v1");
        hash_contract_field(&mut digest, FEATURE_ID_MANIFEST_ID.as_bytes());
        hash_contract_field(&mut digest, FEATURE_LAYOUT_ID.as_bytes());
        for spec in self.specs() {
            hash_contract_field(&mut digest, spec.id.to_string().as_bytes());
            hash_contract_field(&mut digest, feature_role_name(spec.role).as_bytes());
            hash_contract_field(&mut digest, &(spec.channels as u64).to_le_bytes());
            hash_contract_field(&mut digest, spec.scope.as_bytes());
            hash_contract_field(&mut digest, spec.producer_version.as_bytes());
            hash_contract_field(&mut digest, &(spec.dependencies.len() as u64).to_le_bytes());
            for dependency in &spec.dependencies {
                hash_contract_field(&mut digest, dependency.as_bytes());
            }
            hash_contract_field(&mut digest, spec.exactness.as_bytes());
            hash_contract_field(&mut digest, spec.cost_tier.as_bytes());
            hash_contract_field(&mut digest, &(spec.wire_roles.len() as u64).to_le_bytes());
            for role in &spec.wire_roles {
                hash_contract_field(&mut digest, role.as_bytes());
            }
            hash_contract_field(&mut digest, &[u8::from(spec.requires_solver_assisted)]);
            let symmetry = self
                .providers
                .get(&spec.id)
                .map(|provider| provider.symmetry_contract())
                .unwrap_or_else(|| {
                    if structural::is_builtin(spec.id) {
                        "exact_relabel"
                    } else {
                        "unavailable"
                    }
                });
            hash_contract_field(&mut digest, symmetry.as_bytes());
        }
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn feature_role_name(role: FeatureRole) -> &'static str {
    match role {
        FeatureRole::Node => "node",
        FeatureRole::Pair => "pair",
        FeatureRole::Global => "global",
        FeatureRole::Group => "group",
    }
}

fn hash_contract_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}
