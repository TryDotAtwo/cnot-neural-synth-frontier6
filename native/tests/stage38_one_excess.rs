use cnot103_native::{
    recognize_stage38, verify_stage38_result, CnotAlphabet, MatrixState, Stage38Result,
};
use std::collections::{HashMap, VecDeque};

fn bfs_distances(alphabet: &CnotAlphabet) -> HashMap<MatrixState, u64> {
    let identity = MatrixState::identity(alphabet.dimension(), alphabet.id()).unwrap();
    let mut distances = HashMap::from([(identity.clone(), 0)]);
    let mut queue = VecDeque::from([identity]);
    while let Some(state) = queue.pop_front() {
        let next_distance = distances[&state] + 1;
        for gate in alphabet.gates().iter().copied() {
            let mut neighbor = state.clone();
            neighbor.apply_cnot(gate).unwrap();
            if !distances.contains_key(&neighbor) {
                distances.insert(neighbor.clone(), next_distance);
                queue.push_back(neighbor);
            }
        }
    }
    distances
}

fn changed_rows(state: &MatrixState) -> usize {
    (0..state.dimension())
        .filter(|row| {
            state
                .row_words(*row)
                .unwrap()
                .iter()
                .enumerate()
                .any(|(word, value)| {
                    *value
                        != if word == (*row as usize / 64) {
                            1u64 << (*row % 64)
                        } else {
                            0
                        }
                })
        })
        .count()
}

#[test]
fn stage38_matches_all_gl3_distance_partitions_and_verifies_exact_words() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let distances = bfs_distances(&alphabet);
    assert_eq!(distances.len(), 168);
    let mut exact = 0;
    let mut excluded = 0;
    let mut unsupported = 0;
    for (state, distance) in distances {
        let row_bound = changed_rows(&state) as u64;
        let result = recognize_stage38(&state, &alphabet, 100_000).unwrap();
        match (&result, distance) {
            (Stage38Result::Exact(_), d) if d == row_bound + 1 => {
                assert!(verify_stage38_result(&result, &alphabet).unwrap());
                exact += 1;
            }
            (Stage38Result::CertifiedNoSolutionWithinScope(_), d) if d > row_bound + 1 => {
                excluded += 1;
            }
            (Stage38Result::Unsupported(_), d) if d == row_bound => unsupported += 1,
            (Stage38Result::ResourceLimit(_), _) => {
                panic!("GL(3,2) unexpectedly exceeded the explicit contraction budget")
            }
            _ => panic!("stage-38 status disagrees with exact GL(3,2) distance"),
        }
    }
    assert_eq!((exact, excluded, unsupported), (75, 35, 58));
}

#[test]
fn stage38_reports_budget_exhaustion_without_a_negative_claim() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let swap = MatrixState::from_packed_rows(2, alphabet.id(), vec![2, 1]).unwrap();
    let result = recognize_stage38(&swap, &alphabet, 1).unwrap();
    assert!(matches!(result, Stage38Result::ResourceLimit(_)));
}
