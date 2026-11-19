use super::super::StateAdmission;
use super::Validated;
use crate::alphabet::CnotAlphabet;
use crate::packet::StateIdentity;
use crate::training_data::{elementary_lower_bound, ExactCatalog, TrainingRecord};
use std::collections::{HashMap, HashSet};

// The device constructs its own successors. Only the CPU loss needs to keep them.
// An external policy must still see every gate, in alphabet order, then the root.
pub(super) fn admitted_upper_bound(
    alphabet: &CnotAlphabet,
    exact: Option<&ExactCatalog>,
    forbidden: &HashSet<StateIdentity>,
    record: &TrainingRecord,
    admission: Option<&dyn StateAdmission>,
) -> Result<u64, String> {
    let upper = record
        .validate_for_training(alphabet, exact)
        .map_err(|e| e.to_string())?;
    let root = &record.state;
    if admission.is_none() && forbidden.len() < alphabet.gates().len() {
        for other in forbidden {
            if other.dimension == root.dimension()
                && other.alphabet_id == root.alphabet_id()
                && root
                    .gate_to_rows(&other.packed_rows)
                    .is_some_and(|gate| alphabet.cost(gate).is_some())
            {
                return Err("forbidden state in complete backup; reject whole parent".into());
            }
        }
    } else {
        let mut next = root.clone();
        for &gate in alphabet.gates() {
            next.apply_cnot(gate).map_err(|e| e.to_string())?;
            if !forbidden.is_empty() && forbidden.contains(&StateIdentity::from(&next)) {
                return Err("forbidden state in complete backup; reject whole parent".into());
            }
            if let Some(check) = admission {
                check.admit(&next)?;
            }
            next.apply_cnot(gate).map_err(|e| e.to_string())?;
        }
    }
    if !forbidden.is_empty() && forbidden.contains(&StateIdentity::from(root)) {
        return Err("forbidden state in training batch".into());
    }
    if let Some(check) = admission {
        check.admit(root)?;
    }
    Ok(upper)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::WordCertificate;
    use crate::state::MatrixState;
    use crate::training_data::TrainingSplit;
    use crate::CnotGate;
    use std::cell::RefCell;

    #[derive(Default)]
    struct LogAdmission(RefCell<Vec<MatrixState>>);

    impl StateAdmission for LogAdmission {
        fn contract_sha256(&self) -> &str {
            "fixture"
        }
        fn admit(&self, s: &MatrixState) -> Result<(), String> {
            self.0.borrow_mut().push(s.clone());
            Ok(())
        }
    }

    #[test]
    fn device_preflight_matches_reference_for_subsets_and_external_callbacks() {
        let a = CnotGate {
            control: 0,
            target: 1,
        };
        let b = CnotGate {
            control: 64,
            target: 0,
        };
        let alphabet = CnotAlphabet::new(65, "two-gate", vec![a, b], vec![1, 1]).unwrap();
        let root = MatrixState::identity(65, alphabet.id()).unwrap();
        let record = TrainingRecord {
            state: root.clone(),
            split: TrainingSplit::Train,
            lineage_id: "root".into(),
            exact_distance: Some(0),
            verified_words: vec![WordCertificate {
                start: root.clone(),
                end: root.clone(),
                gates: vec![],
                physical_cost: 0,
                producer: "fixture".into(),
                lineage_id: "root".into(),
            }],
        };
        let mut allowed_neighbors = Vec::new();
        for &gate in alphabet.gates() {
            let mut next = root.clone();
            next.apply_cnot(gate).unwrap();
            allowed_neighbors.push(next);
        }
        let mut outside = root.clone();
        outside
            .apply_cnot(CnotGate {
                control: 1,
                target: 2,
            })
            .unwrap();
        let alphabets = HashMap::from([(alphabet.id().to_owned(), alphabet.clone())]);
        for forbidden in [
            vec![],
            vec![&outside],
            vec![&allowed_neighbors[1]],
            vec![&root],
            vec![&outside, &allowed_neighbors[0]],
        ] {
            let keys: HashSet<_> = forbidden.into_iter().map(StateIdentity::from).collect();
            let full = validate_batch(
                &alphabets,
                &HashMap::new(),
                &keys,
                std::slice::from_ref(&record),
                None,
            )
            .map(|rows| rows[0].upper_bound);
            assert_eq!(
                admitted_upper_bound(&alphabet, None, &keys, &record, None),
                full
            );
        }
        let log = LogAdmission::default();
        assert_eq!(
            admitted_upper_bound(&alphabet, None, &HashSet::new(), &record, Some(&log)).unwrap(),
            0
        );
        allowed_neighbors.push(root);
        assert_eq!(*log.0.borrow(), allowed_neighbors);
    }
}
