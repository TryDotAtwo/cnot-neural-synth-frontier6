use cnot103_native::{
    CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, MatrixState,
};

#[test]
fn g05_is_exact_ordered_pair_column_bit_histogram() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();

    let id = FeatureId::new(b'G', 5).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "g05-exact-pair-histogram-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = FeatureRegistry::new().build(&[state], &policy).unwrap();
    let feature = packet.feature(id);
    assert!(feature.row_is_available(0));

    let offset = |left: usize, right: usize, channel: usize| (left * 3 + right) * 4 + channel;
    assert_eq!(
        &feature.values[offset(0, 1, 0)..offset(0, 1, 0) + 4],
        &[1.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(
        &feature.values[offset(1, 0, 0)..offset(1, 0, 0) + 4],
        &[1.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        &feature.values[offset(0, 0, 0)..offset(0, 0, 0) + 4],
        &[1.0, 0.0, 0.0, 2.0]
    );
}
