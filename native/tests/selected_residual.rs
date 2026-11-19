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

#[test]
fn multi_width_selected_residual_dispatch_is_bound_to_each_physical_alphabet() {
    use cnot103_native::{register_selected_residual_ranks_multi, SelectedResidualContext};
    let a3 = CnotAlphabet::all_to_all_unit(3).unwrap();
    let a4 = CnotAlphabet::all_to_all_unit(4).unwrap();
    let c3 = SelectedResidualContext { alphabet: a3.clone(), gates: vec![a3.gates()[0], a3.gates()[1]] };
    let c4 = SelectedResidualContext { alphabet: a4.clone(), gates: vec![a4.gates()[2]] };
    let mut registry = FeatureRegistry::new();
    register_selected_residual_ranks_multi(&mut registry, vec![c4.clone(), c3.clone()]).unwrap();
    let mut reordered = FeatureRegistry::new();
    register_selected_residual_ranks_multi(&mut reordered, vec![c3.clone(), c4.clone()]).unwrap();
    assert_eq!(registry.contract_sha256(), reordered.contract_sha256());
    let mut changed = FeatureRegistry::new();
    let mut other4 = c4.clone();
    other4.gates = vec![a4.gates()[0]];
    register_selected_residual_ranks_multi(&mut changed, vec![c3.clone(), other4]).unwrap();
    assert_ne!(registry.contract_sha256(), changed.contract_sha256());
    assert!(register_selected_residual_ranks_multi(&mut FeatureRegistry::new(), vec![c3.clone(), c3.clone()]).is_err());
    let mut invalid = c4.clone();
    invalid.gates = vec![CnotGate { control: 0, target: 0 }];
    assert!(register_selected_residual_ranks_multi(&mut FeatureRegistry::new(), vec![c3, invalid]).is_err());
    let policy = FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'C', 10).unwrap()],
        seed: 0,
        schema_id: cnot103_native::schema::MISSING_TOKEN_LAYOUT_ID.into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    for (alphabet, count) in [(&a3, 2), (&a4, 1)] {
        let state = MatrixState::identity(alphabet.dimension(), alphabet.id()).unwrap();
        let packet = registry.build(&[state], &policy).unwrap();
        let feature = packet.feature(FeatureId::new(b'C', 10).unwrap());
        assert_eq!(feature.status, vec![cnot103_native::FeatureStatus::Computed]);
        assert_eq!(feature.values, vec![1.0; count]);
    }
    for state in [MatrixState::identity(3, "wrong-id").unwrap(),
                  MatrixState::identity(5, "no-profile").unwrap()] {
        let packet = registry.build(&[state], &policy).unwrap();
        assert_eq!(packet.feature(FeatureId::new(b'C', 10).unwrap()).status,
                   vec![cnot103_native::FeatureStatus::Unsupported]);
    }
}
