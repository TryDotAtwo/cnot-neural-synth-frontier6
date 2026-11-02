use cnot103_native::{
    recognize_stage35, verify_stage35_result, CnotAlphabet, CnotGate, MatrixState, Stage35Result,
    STAGE35_THEOREM_SHA256,
};

fn matrix_from_small_rows(n: u32, alphabet_id: &str, rows: &[u64]) -> MatrixState {
    assert_eq!(rows.len(), n as usize);
    MatrixState::from_packed_rows(n, alphabet_id, rows.to_vec()).unwrap()
}

fn generated_stage35_state(
    n: u32,
    alphabet: &CnotAlphabet,
    frame: [u8; 2],
    external_vectors: &[u8],
) -> MatrixState {
    let mut rows: Vec<u64> = (0..n).map(|index| 1u64 << index).collect();
    rows[0] = (rows[0] & !3) | u64::from(frame[0]);
    rows[1] = (rows[1] & !3) | u64::from(frame[1]);
    for (offset, vector) in external_vectors.iter().copied().enumerate() {
        let column = offset + 2;
        let values = [
            u8::from((frame[0] & vector).count_ones() & 1 != 0),
            u8::from((frame[1] & vector).count_ones() & 1 != 0),
        ];
        rows[0] |= u64::from(values[0]) << column;
        rows[1] |= u64::from(values[1]) << column;
    }
    matrix_from_small_rows(n, alphabet.id(), &rows)
}

#[test]
fn stage35_exact_oracle_covers_all_two_wire_frames_and_verifies_words() {
    let n = 5;
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let frames = [[1, 2], [1, 3], [3, 2], [2, 3], [3, 1], [2, 1]];

    for frame in frames {
        for pattern in 0usize..64 {
            let vectors: Vec<u8> = (0..3)
                .map(|offset| ((pattern >> (2 * offset)) & 3) as u8)
                .collect();
            let state = generated_stage35_state(n, &alphabet, frame, &vectors);
            let result = recognize_stage35(&state, &alphabet).unwrap();
            let Stage35Result::Exact(exact) = &result else {
                panic!("stage-35-generated state was not recognized: {result:?}");
            };
            assert_eq!(exact.theorem_sha256, STAGE35_THEOREM_SHA256);
            assert_eq!(exact.distance, exact.lower_bound);
            assert!(verify_stage35_result(&result, &alphabet).unwrap());
        }
    }
}

#[test]
fn stage35_returns_unsupported_without_an_exact_label_outside_its_scope() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    for gate in [
        CnotGate {
            control: 0,
            target: 1,
        },
        CnotGate {
            control: 1,
            target: 2,
        },
        CnotGate {
            control: 2,
            target: 0,
        },
    ] {
        state.apply_cnot(gate).unwrap();
    }

    let result = recognize_stage35(&state, &alphabet).unwrap();
    let Stage35Result::Unsupported(unsupported) = &result else {
        panic!("three-changed-row state must remain unsupported");
    };
    assert_eq!(unsupported.reason, "more than two changed rows");
    assert!(!verify_stage35_result(&result, &alphabet).unwrap());
}

#[test]
fn stage35_rejects_weighted_alphabet_for_exact_labels() {
    let dimension = 3;
    let gates: Vec<_> = (0..dimension)
        .flat_map(|control| {
            (0..dimension)
                .filter(move |target| *target != control)
                .map(move |target| CnotGate { control, target })
        })
        .collect();
    let alphabet = CnotAlphabet::new(
        dimension,
        "weighted-full-cnot",
        gates.clone(),
        vec![2; gates.len()],
    )
    .unwrap();
    let state = MatrixState::identity(dimension, alphabet.id()).unwrap();

    let result = recognize_stage35(&state, &alphabet).unwrap();
    let Stage35Result::Unsupported(unsupported) = &result else {
        panic!("weighted alphabet must not receive a unit-cost exact label");
    };
    assert_eq!(unsupported.reason, "requires full unit CNOT alphabet");
}
