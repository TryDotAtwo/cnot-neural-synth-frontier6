use cnot103_native::{
    CnotAlphabet, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole, FeatureStatus,
    MatrixState,
};

#[test]
fn d03_matches_the_source_ordered_row_pair_histogram() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let state = MatrixState::from_packed_rows(4, alphabet.id(), vec![1, 3, 6, 12]).unwrap();
    let id = FeatureId::new(b'D', 3).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "d03-source-row-pair-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };

    let packet = FeatureRegistry::new().build(&[state], &policy).unwrap();
    let feature = packet.feature(id);
    assert_eq!(feature.spec.role, FeatureRole::Pair);
    assert_eq!(feature.spec.channels, 3);
    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    // Source row pair (0, 1): bins 10, 01, 11 among the four columns.
    assert_eq!(&feature.values[3..6], &[0.0, 1.0, 1.0]);
    // Ordered reverse pair (1, 0) swaps the exclusive bins.
    assert_eq!(&feature.values[12..15], &[1.0, 0.0, 1.0]);
}
