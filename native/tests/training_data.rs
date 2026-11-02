use cnot103_native::training_data::{
    ExactCatalog, ExactCatalogEntry, TrainingRecord, TrainingSplit,
};
use cnot103_native::{CnotAlphabet, CnotGate, MatrixState, WordCertificate};
use std::collections::{HashMap, VecDeque};

fn exact_gl(alphabet: &CnotAlphabet) -> ExactCatalog {
    let identity = MatrixState::identity(alphabet.dimension(), alphabet.id()).unwrap();
    let mut paths = HashMap::from([(identity.clone(), Vec::<CnotGate>::new())]);
    let mut queue = VecDeque::from([identity.clone()]);
    while let Some(state) = queue.pop_front() {
        let path = paths[&state].clone();
        for gate in alphabet.gates().iter().copied() {
            let mut successor = state.clone();
            successor.apply_cnot(gate).unwrap();
            if !paths.contains_key(&successor) {
                let mut successor_path = path.clone();
                successor_path.push(gate);
                paths.insert(successor.clone(), successor_path);
                queue.push_back(successor);
            }
        }
    }

    let entries = paths
        .into_iter()
        .map(|(state, path)| {
            let gates: Vec<_> = path.into_iter().rev().collect();
            let distance = gates.len() as u64;
            let certificate = WordCertificate {
                start: state.clone(),
                end: identity.clone(),
                gates,
                physical_cost: distance,
                producer: "test_bfs".into(),
                lineage_id: format!("gl2:{distance}"),
            };
            (
                state,
                ExactCatalogEntry {
                    distance,
                    certificate,
                },
            )
        })
        .collect();
    ExactCatalog::new(alphabet, entries).unwrap()
}

#[test]
fn exact_catalog_requires_a_closed_component_and_proves_distances() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let catalog = exact_gl(&alphabet);
    let identity = MatrixState::identity(2, alphabet.id()).unwrap();
    assert_eq!(catalog.exact_distance(&identity), Some(0));
    assert_eq!(catalog.state_count(), 6);
    assert_eq!(catalog.alphabet_id(), alphabet.id());
    assert_eq!(catalog.sha256().len(), 64);

    let mut incomplete = catalog.entries().clone();
    let omitted = incomplete
        .keys()
        .find(|state| **state != identity)
        .unwrap()
        .clone();
    incomplete.remove(&omitted);
    assert!(ExactCatalog::new(&alphabet, incomplete)
        .unwrap_err()
        .to_string()
        .contains("not closed"));
}

#[test]
fn training_record_requires_verified_reduction_and_catalog_backed_exact_label() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let catalog = exact_gl(&alphabet);
    let state = catalog
        .entries()
        .keys()
        .find(|state| !state.is_identity())
        .unwrap()
        .clone();
    let exact = catalog.exact_distance(&state).unwrap();
    let certificate = catalog.entries()[&state].certificate.clone();
    let record = TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![certificate.clone()],
        exact_distance: Some(exact),
        lineage_id: "fixture-lineage".into(),
    };
    assert_eq!(
        record
            .validate_for_training(&alphabet, Some(&catalog))
            .unwrap(),
        exact
    );

    let mut bad_label = record.clone();
    bad_label.exact_distance = Some(exact + 1);
    assert!(bad_label
        .validate_for_training(&alphabet, Some(&catalog))
        .is_err());

    let mut bad_word = record.clone();
    bad_word.verified_words[0].physical_cost += 1;
    assert!(bad_word
        .validate_for_training(&alphabet, Some(&catalog))
        .is_err());

    let mut held_out = record;
    held_out.split = TrainingSplit::Test;
    assert!(held_out
        .validate_for_training(&alphabet, Some(&catalog))
        .unwrap_err()
        .to_string()
        .contains("train split"));
}

#[test]
fn exact_label_can_be_proved_by_tight_verified_bounds() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let identity = MatrixState::identity(2, alphabet.id()).unwrap();
    let gate = alphabet.gates()[0];
    let mut state = identity.clone();
    state.apply_cnot(gate).unwrap();
    let exact = cnot103_native::training_data::elementary_lower_bound(&state).unwrap();
    assert_eq!(exact, 1);
    let certificate = WordCertificate {
        start: state.clone(),
        end: identity,
        gates: vec![gate],
        physical_cost: 1,
        producer: "fixture_exact_reduction".into(),
        lineage_id: "fixture-tight-bound".into(),
    };
    let record = TrainingRecord {
        state: state.clone(),
        split: TrainingSplit::Train,
        verified_words: vec![certificate.clone()],
        exact_distance: Some(exact),
        lineage_id: "train-tight-bound".into(),
    };
    assert_eq!(
        record.validate_for_training(&alphabet, None).unwrap(),
        exact
    );

    let mut loose = record;
    let mut word = certificate;
    word.gates.splice(0..0, [gate, gate]);
    word.physical_cost += 2;
    loose.verified_words = vec![word];
    assert!(loose.validate_for_training(&alphabet, None).is_err());
}

#[test]
fn source_elementary_lower_bound_never_exceeds_exact_gl3_distance() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let catalog = exact_gl(&alphabet);
    assert_eq!(catalog.state_count(), 168);
    for (state, entry) in catalog.entries() {
        let lower = cnot103_native::training_data::elementary_lower_bound(state).unwrap();
        assert!(
            lower <= entry.distance,
            "lower bound {lower} exceeds exact distance {} for state {:?}",
            entry.distance,
            state.packed_rows()
        );
    }
}
