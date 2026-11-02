use cnot103_native::{
    recognize_stage37, verify_stage37_result, CnotAlphabet, CnotGate, MatrixState, Stage37Error,
    Stage37Result,
};
use std::collections::{HashMap, VecDeque};

fn exact_distances(alphabet: &CnotAlphabet) -> HashMap<MatrixState, u64> {
    let identity = MatrixState::identity(alphabet.dimension(), alphabet.id()).unwrap();
    let mut distances = HashMap::from([(identity.clone(), 0)]);
    let mut queue = VecDeque::from([identity]);
    while let Some(state) = queue.pop_front() {
        let distance = distances[&state];
        for gate in alphabet.gates().iter().copied() {
            let mut successor = state.clone();
            successor.apply_cnot(gate).unwrap();
            if !distances.contains_key(&successor) {
                distances.insert(successor.clone(), distance + 1);
                queue.push_back(successor);
            }
        }
    }
    distances
}

fn changed_rows(state: &MatrixState) -> usize {
    (0..state.dimension())
        .filter(|row| {
            let mut identity_row = vec![0u64; state.words_per_row()];
            identity_row[*row as usize / 64] = 1u64 << (*row as usize % 64);
            state.row_words(*row).unwrap() != identity_row
        })
        .count()
}

#[test]
fn stage37_matches_full_bfs_over_gl_2_3_and_4() {
    for (dimension, expected_states) in [(2, 6), (3, 168), (4, 20_160)] {
        let alphabet = CnotAlphabet::all_to_all_unit(dimension).unwrap();
        let distances = exact_distances(&alphabet);
        assert_eq!(distances.len(), expected_states);
        let mut exact_count = 0usize;
        for (state, exact_distance) in &distances {
            let lower_bound = changed_rows(state) as u64;
            match recognize_stage37(state, &alphabet).unwrap() {
                Stage37Result::Exact(exact) => {
                    exact_count += 1;
                    assert_eq!(exact.distance, lower_bound);
                    assert_eq!(*exact_distance, lower_bound);
                    assert_eq!(exact.word.start, *state);
                    assert!(
                        verify_stage37_result(&Stage37Result::Exact(exact), &alphabet).unwrap()
                    );
                }
                Stage37Result::NoZeroExcess(no_solution) => {
                    assert_eq!(no_solution.excluded_length, lower_bound);
                    assert!(*exact_distance > lower_bound);
                    assert!(!no_solution.missing_rows.is_empty());
                }
                Stage37Result::Unsupported(unsupported) => {
                    panic!("full unit alphabet was rejected: {unsupported:?}")
                }
            }
        }
        assert!(exact_count > 0);
    }
}

#[test]
fn stage37_constructs_and_replays_words_beyond_one_machine_word() {
    for dimension in [8u32, 13, 25, 34, 50, 72, 90, 108, 122, 144] {
        let alphabet = CnotAlphabet::all_to_all_unit(dimension).unwrap();
        let mut state = MatrixState::identity(dimension, alphabet.id()).unwrap();
        for target in 1..dimension {
            let control = (target * 17 + 3) % target;
            state.apply_cnot(CnotGate { control, target }).unwrap();
        }
        let result = recognize_stage37(&state, &alphabet).unwrap();
        let Stage37Result::Exact(exact) = &result else {
            panic!("constructed word was not recognized exactly: {result:?}");
        };
        assert_eq!(exact.distance, u64::from(dimension - 1));
        assert_eq!(exact.changed_rows.len(), dimension as usize - 1);
        assert!(verify_stage37_result(&result, &alphabet).unwrap());
    }
}

#[test]
fn stage37_nonexact_result_excludes_only_the_changed_row_bound() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let state = MatrixState::from_packed_rows(2, alphabet.id(), vec![0b10, 0b01]).unwrap();
    let result = recognize_stage37(&state, &alphabet).unwrap();
    let Stage37Result::NoZeroExcess(no_solution) = result else {
        panic!("wire swap should not saturate its two-row lower bound");
    };
    assert_eq!(no_solution.excluded_length, 2);
    assert_eq!(no_solution.missing_rows, vec![0, 1]);
    assert!(!verify_stage37_result(&Stage37Result::NoZeroExcess(no_solution), &alphabet).unwrap());
}

#[test]
fn stage37_rejects_nonunit_and_noncomplete_alphabets_without_labels() {
    let gates = vec![
        CnotGate {
            control: 0,
            target: 1,
        },
        CnotGate {
            control: 1,
            target: 0,
        },
    ];
    let weighted = CnotAlphabet::new(2, "weighted", gates.clone(), vec![2, 2]).unwrap();
    let weighted_state = MatrixState::identity(2, weighted.id()).unwrap();
    assert!(matches!(
        recognize_stage37(&weighted_state, &weighted).unwrap(),
        Stage37Result::Unsupported(_)
    ));

    let incomplete = CnotAlphabet::new(2, "incomplete", gates[..1].to_vec(), vec![1]).unwrap();
    let incomplete_state = MatrixState::identity(2, incomplete.id()).unwrap();
    assert!(matches!(
        recognize_stage37(&incomplete_state, &incomplete).unwrap(),
        Stage37Result::Unsupported(_)
    ));
}

#[test]
fn stage37_handles_identity_and_rejects_alphabet_mismatch() {
    let singleton = CnotAlphabet::all_to_all_unit(1).unwrap();
    let identity = MatrixState::identity(1, singleton.id()).unwrap();
    let result = recognize_stage37(&identity, &singleton).unwrap();
    let Stage37Result::Exact(exact) = &result else {
        panic!("identity must attain the zero changed-row bound");
    };
    assert_eq!(exact.distance, 0);
    assert!(exact.word.gates.is_empty());
    assert!(verify_stage37_result(&result, &singleton).unwrap());

    let other = CnotAlphabet::all_to_all_unit(2).unwrap();
    assert_eq!(
        recognize_stage37(&identity, &other),
        Err(Stage37Error::AlphabetMismatch)
    );
}
