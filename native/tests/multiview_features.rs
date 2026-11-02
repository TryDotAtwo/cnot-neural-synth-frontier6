use cnot103_native::{
    CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole, MatrixState,
};

#[test]
fn h05_aligns_the_four_exact_matrix_views_per_ordered_wire_pair() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();

    let id = FeatureId::new(b'H', 5).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "h05-aligned-matrix-views-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = FeatureRegistry::new().build(&[state], &policy).unwrap();
    let feature = packet.feature(id);

    assert!(feature.row_is_available(0));
    assert_eq!(feature.spec.role, FeatureRole::Pair);
    assert_eq!(feature.spec.channels, 4);
    assert_eq!(
        feature.spec.producer_version,
        "aligned_four_matrix_views_pair_tokens_v1"
    );
    assert_eq!(
        feature.spec.dependencies,
        vec!["M", "transpose", "inverse", "inverse_transpose"]
    );
    assert_eq!(feature.shape, vec![1, 3, 3, 4]);

    let offset = |row: usize, column: usize| (row * 3 + column) * 4;
    assert_eq!(
        &feature.values[offset(0, 1)..offset(0, 1) + 4],
        &[0.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(
        &feature.values[offset(1, 0)..offset(1, 0) + 4],
        &[1.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        &feature.values[offset(0, 0)..offset(0, 0) + 4],
        &[1.0, 1.0, 1.0, 1.0]
    );
}
