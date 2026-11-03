use super::super::StateAdmission;
use super::Validated;
use crate::alphabet::CnotAlphabet;
use crate::packet::StateIdentity;
use crate::training_data::{elementary_lower_bound, ExactCatalog, TrainingRecord};
use std::collections::{HashMap, HashSet};

pub(super) fn validate_batch<'a>(
    alphabets: &'a HashMap<String, CnotAlphabet>,
    exact_catalogs: &HashMap<String, ExactCatalog>,
    forbidden: &HashSet<StateIdentity>,
    records: &'a [TrainingRecord],
    admission: Option<&dyn StateAdmission>,
) -> Result<Vec<Validated<'a>>, String> {
    // Finish all split, certificate, bound, forbidden-neighbor, and
    // external-admission checks before mutating exposures or calling the model.
    let mut validated = Vec::with_capacity(records.len());
    for record in records {
        let alphabet = alphabets
            .get(record.state.alphabet_id())
            .ok_or_else(|| "training record uses an unregistered alphabet".to_owned())?;
        let exact = exact_catalogs.get(alphabet.id());
        let upper_bound = record
            .validate_for_training(alphabet, exact)
            .map_err(|error| error.to_string())?;
        let lower_bound =
            elementary_lower_bound(&record.state).map_err(|error| error.to_string())?;
        if lower_bound > upper_bound {
            return Err("verified word upper bound is below the source lower bound".into());
        }

        let mut successors = Vec::with_capacity(alphabet.gates().len());
        for gate in alphabet.gates().iter().copied() {
            let mut successor = record.state.clone();
            successor
                .apply_cnot(gate)
                .map_err(|error| error.to_string())?;
            if forbidden.contains(&StateIdentity::from(&successor)) {
                return Err("forbidden state in complete backup; reject whole parent".into());
            }
            if let Some(admission) = admission {
                admission.admit(&successor)?;
            }
            successors.push(successor);
        }
        if forbidden.contains(&StateIdentity::from(&record.state)) {
            return Err("forbidden state in training batch".into());
        }
        if let Some(admission) = admission {
            admission.admit(&record.state)?;
        }
        validated.push(Validated {
            record,
            alphabet,
            lower_bound,
            upper_bound,
            successors,
        });
    }

    Ok(validated)
}
