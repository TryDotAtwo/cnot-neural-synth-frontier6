use cnot103_native::{
    register_partial_shear_rank, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureStatus, MatrixState,
};

fn policy(symmetry_contract: &str) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: [FeatureId::new(b'C', 5).unwrap()].into_iter().collect(),
        seed: 0,
        schema_id: "partial-shear-rank-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes: None,
    }
}

#[test]
fn partial_shear_feature_ranks_the_source_bound_elimination_block() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    let mut registry = FeatureRegistry::new();
    register_partial_shear_rank(&mut registry).unwrap();

    let packet = registry
        .build(&[state], &policy("augmentation_only"))
        .unwrap();
    let feature = packet.feature(FeatureId::new(b'C', 5).unwrap());
    assert!(feature.row_is_available(0));
    assert_eq!(feature.values, vec![1.0]);
}

#[test]
fn partial_shear_feature_marks_zero_block_and_nonaugmentation_policy_unsupported() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let identity = MatrixState::identity(4, alphabet.id()).unwrap();
    let mut registry = FeatureRegistry::new();
    register_partial_shear_rank(&mut registry).unwrap();

    let packet = registry
        .build(&[identity.clone()], &policy("augmentation_only"))
        .unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'C', 5).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );
    let packet = registry.build(&[identity], &policy("exact")).unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'C', 5).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );
}
