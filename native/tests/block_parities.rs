use cnot103_native::{
    register_block_parities, BlockContext, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureStatus, MatrixState,
};

#[test]
fn block_parity_features_match_hand_counted_patterns_and_roles() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    for gate in [
        CnotGate {
            control: 0,
            target: 2,
        },
        CnotGate {
            control: 1,
            target: 2,
        },
        CnotGate {
            control: 1,
            target: 3,
        },
    ] {
        state.apply_cnot(gate).unwrap();
    }
    let context = BlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_block_parities(&mut registry, vec![context], 8, 8).unwrap();
    let ids = [1, 2, 4, 5, 6, 7, 8]
        .into_iter()
        .map(|ordinal| FeatureId::new(b'D', ordinal).unwrap())
        .collect();
    let policy = FeaturePolicy {
        enabled_ids: ids,
        seed: 0,
        schema_id: "block-parities-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };

    let packet = registry.build(&[state], &policy).unwrap();
    for (ordinal, expected) in [
        (1, vec![0.0, 1.0, 1.0]),
        (2, vec![1.0; 3]),
        (4, vec![0.0, 1.0, 1.0]),
    ] {
        let feature = packet.feature(FeatureId::new(b'D', ordinal).unwrap());
        assert_eq!(feature.values, expected);
        assert_eq!(feature.token_valid, Some(vec![1; 3]));
    }
    assert_eq!(
        packet
            .feature(FeatureId::new(b'D', 1).unwrap())
            .incidence
            .as_ref()
            .unwrap()[..12],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 5).unwrap()).values,
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 6).unwrap()).values,
        vec![1.0]
    );
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 6).unwrap()).incidence,
        Some(vec![
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    );
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 7).unwrap()).values,
        vec![2.0, 2.0, 2.0, 0.0, 1.0]
    );
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 8).unwrap()).values,
        vec![3.0]
    );
}

#[test]
fn block_parity_provider_enforces_pattern_budget_and_augmentation_policy() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    state
        .apply_cnot(CnotGate {
            control: 1,
            target: 3,
        })
        .unwrap();
    let context = BlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_block_parities(&mut registry, vec![context.clone()], 2, 8).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: [FeatureId::new(b'D', 1).unwrap()].into_iter().collect(),
        seed: 0,
        schema_id: "block-parities-budget-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&[state.clone()], &policy).unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 1).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let mut registry = FeatureRegistry::new();
    register_block_parities(&mut registry, vec![context], 8, 8).unwrap();
    let mut exact_policy = policy;
    exact_policy.symmetry_contract = "exact".into();
    let packet = registry.build(&[state], &exact_policy).unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'D', 1).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );
}
