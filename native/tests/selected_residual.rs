use cnot103_native::{
    register_selected_residual_ranks, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, MatrixState,
};

#[test]
fn selected_residual_rank_uses_the_physical_alphabet_and_gate() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, alphabet.id()).unwrap();
    let selected = CnotGate {
        control: 0,
        target: 1,
    };
    let mut registry = FeatureRegistry::new();
    register_selected_residual_ranks(&mut registry, &alphabet, vec![selected]).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: [FeatureId::new(b'C', 10).unwrap()].into_iter().collect(),
        seed: 0,
        schema_id: "selected-residual-rank-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };

    let packet = registry.build(&[state], &policy).unwrap();
    let feature = packet.feature(FeatureId::new(b'C', 10).unwrap());
    assert!(feature.row_is_available(0));
    assert_eq!(feature.values, vec![1.0]);
    assert_eq!(feature.token_valid, Some(vec![1]));
    assert_eq!(feature.incidence, Some(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
}

#[test]
fn selected_residual_rank_rejects_an_alphabet_mismatch() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, "other-alphabet").unwrap();
    let selected = CnotGate {
        control: 0,
        target: 1,
    };
    let mut registry = FeatureRegistry::new();
    register_selected_residual_ranks(&mut registry, &alphabet, vec![selected]).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: [FeatureId::new(b'C', 10).unwrap()].into_iter().collect(),
        seed: 0,
        schema_id: "selected-residual-rank-mismatch-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };

    let packet = registry.build(&[state], &policy).unwrap();
    let feature = packet.feature(FeatureId::new(b'C', 10).unwrap());
    assert!(!feature.row_is_available(0));
    assert_eq!(
        feature.status,
        vec![cnot103_native::FeatureStatus::Unsupported]
    );
}
