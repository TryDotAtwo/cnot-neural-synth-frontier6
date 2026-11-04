use super::FullGraphObjectiveConfig;
use crate::alphabet::CnotAlphabet;
use crate::packet::StateIdentity;
use crate::state::MatrixState;
use crate::training_data::{ExactCatalog, TrainingRecord};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

pub(super) struct ObjectiveFingerprintInput<'a> {
    pub(super) alphabets: &'a HashMap<String, CnotAlphabet>,
    pub(super) catalogs: &'a HashMap<String, ExactCatalog>,
    pub(super) policy_id: &'a str,
    pub(super) feature_contract_sha256: &'a str,
    pub(super) forbidden: &'a HashSet<StateIdentity>,
    pub(super) config: &'a FullGraphObjectiveConfig,
    pub(super) training_records_sha256: Option<&'a str>,
    pub(super) frozen_dataset_sha256: Option<&'a str>,
}

pub(super) fn validate_identity(
    identity: &StateIdentity,
    alphabets: &HashMap<String, CnotAlphabet>,
) -> Result<(), String> {
    let alphabet = alphabets
        .get(&identity.alphabet_id)
        .ok_or_else(|| "exposure identity references an unknown alphabet".to_owned())?;
    if identity.dimension != alphabet.dimension() {
        return Err("exposure identity dimension differs from alphabet".into());
    }
    MatrixState::from_packed_rows(
        identity.dimension,
        identity.alphabet_id.clone(),
        identity.packed_rows.clone(),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

pub(super) fn sorted_identities(states: &HashSet<StateIdentity>) -> Vec<StateIdentity> {
    let mut result: Vec<_> = states.iter().cloned().collect();
    result.sort_by_key(identity_bytes);
    result
}

pub(super) fn identity_set_digest(states: &HashSet<StateIdentity>) -> String {
    let mut digest = Sha256::new();
    let mut rows: Vec<_> = states.iter().map(identity_bytes).collect();
    rows.sort();
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        digest.update((row.len() as u64).to_le_bytes());
        digest.update(row);
    }
    hex_digest(digest.finalize().into())
}

fn identity_bytes(identity: &StateIdentity) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&identity.dimension.to_le_bytes());
    bytes.extend_from_slice(&(identity.alphabet_id.len() as u64).to_le_bytes());
    bytes.extend_from_slice(identity.alphabet_id.as_bytes());
    bytes.extend_from_slice(&(identity.packed_rows.len() as u64).to_le_bytes());
    for row in &identity.packed_rows {
        bytes.extend_from_slice(&row.to_le_bytes());
    }
    bytes
}

pub(super) fn objective_config_digest(input: ObjectiveFingerprintInput<'_>) -> String {
    let ObjectiveFingerprintInput {
        alphabets,
        catalogs,
        policy_id,
        feature_contract_sha256,
        forbidden,
        config,
        training_records_sha256,
        frozen_dataset_sha256,
    } = input;
    let mut alphabet_rows: Vec<_> = alphabets
        .values()
        .map(|alphabet| {
            serde_json::json!({
                "id": alphabet.id(),
                "dimension": alphabet.dimension(),
                "gates": alphabet.gates().iter().map(|gate| [gate.control, gate.target]).collect::<Vec<_>>(),
                "costs": (0..alphabet.gates().len()).map(|index| alphabet.cost_at(index).unwrap_or_default()).collect::<Vec<_>>()
            })
        })
        .collect();
    alphabet_rows.sort_by_key(|row| row["id"].as_str().unwrap_or_default().to_owned());
    let mut catalog_rows: Vec<_> = catalogs
        .iter()
        .map(|(id, catalog)| (id, catalog.sha256()))
        .collect();
    catalog_rows.sort_by_key(|(id, _)| (*id).clone());
    let body = serde_json::json!({
        "id": "full_graph_reference_v3",
        "policy": policy_id,
        "feature_registry_sha256": feature_contract_sha256,
        "frozen_dataset_sha256": frozen_dataset_sha256,
        "training_records_sha256": training_records_sha256,
        "alphabets": alphabet_rows,
        "exact_sources": catalog_rows,
        "forbidden_sha256": identity_set_digest(forbidden),
        "bellman_weight": config.bellman_weight,
        "supervision_weight": config.supervision_weight,
        "qhat_weight": config.qhat_weight,
        "teacher_weight": config.teacher_weight,
        "neighbor_chunk_size": config.neighbor_chunk_size,
        "teacher_temperature": config.teacher_temperature,
        "external_admission": config.external_admission,
        "external_admission_sha256": config.external_admission_sha256,
        "require_all_model_inputs": config.require_all_model_inputs,
        "compact_exposures": config.compact_exposures,
        "gradient_observation": "neural_trainer_autograd_boundary_v1"
    });
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    hex_digest(Sha256::digest(bytes).into())
}

pub(super) fn training_records_digest(records: &[TrainingRecord]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cnot103/ordered_training_records_v1\0");
    digest.update((records.len() as u64).to_le_bytes());
    for record in records {
        let state_identity = StateIdentity::from(&record.state);
        hash_digest_bytes(&mut digest, &identity_bytes(&state_identity));
        digest.update([match record.split {
            crate::training_data::TrainingSplit::Train => 0,
            crate::training_data::TrainingSplit::Dev => 1,
            crate::training_data::TrainingSplit::Test => 2,
        }]);
        hash_digest_bytes(&mut digest, record.lineage_id.as_bytes());
        match record.exact_distance {
            Some(distance) => {
                digest.update([1]);
                digest.update(distance.to_le_bytes());
            }
            None => digest.update([0]),
        }
        digest.update((record.verified_words.len() as u64).to_le_bytes());
        for certificate in &record.verified_words {
            let start = identity_bytes(&StateIdentity::from(&certificate.start));
            let end = identity_bytes(&StateIdentity::from(&certificate.end));
            hash_digest_bytes(&mut digest, &start);
            hash_digest_bytes(&mut digest, &end);
            digest.update((certificate.gates.len() as u64).to_le_bytes());
            for gate in &certificate.gates {
                digest.update(gate.control.to_le_bytes());
                digest.update(gate.target.to_le_bytes());
            }
            digest.update(certificate.physical_cost.to_le_bytes());
            hash_digest_bytes(&mut digest, certificate.producer.as_bytes());
            hash_digest_bytes(&mut digest, certificate.lineage_id.as_bytes());
        }
    }
    hex_digest(digest.finalize().into())
}

fn hash_digest_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hex_digest(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
