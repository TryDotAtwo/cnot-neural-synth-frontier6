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

fn identity_bytes(identu×¯6¶‰Ëkºwµçv—5ö&Æö6µöÆÆö6F–öåöf–ÆVB"çFõö÷væVB‚’“ó°¢f÷"gF&vWB–âf6öçFW‡BçF&vWE÷v—&W2°¢ÆWB×WB&÷rÒfV3£¦æWr‚“°¢&÷rçG'•÷&W6W'fUöW†7B†6öçFW‡Bç6÷W&6U÷v—&W2æÆVâ‚’¢æÖöW'"‡Å÷Â'–Eö&6—5ö&Æö6µöÆÆö6F–öåöf–ÆVB"çFõö÷væVB‚’“ó°¢f÷"g6÷W&6R–âf6öçFW‡Bç6÷W&6U÷v—&W2°¢&÷rçW6‚†ÖG&—…ö&—B‡7FFRÂF&vWBÂ6÷W&6R’“°¢Ğ¢&Æö6²çW6‚‡&÷r“°¢Ğ¢–b&Æö6²æ—FW"‚’æÖ…fV3£¦ÆVâ’ç7VÓ££ÇW6—¦Sâ‚’Ò6VÆÇ2°¢&WGW&âW'"‚'–Eö&6—5ö&Æö6µöW‡G&7F–öåöÖ—6ÖF6‚"æ–çFò‚’“°¢Ğ¢ö²†&Æö6²§Ğ