use cnot103_native::{
    register_basis_gradient, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureStatus, MatrixState,
};

fn policy(ids: &[FeatureId]) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: ids.to_vec(),
        seed: 0,
        schema_id: "basis-gradient-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn id(family: u8, ordinal: u8) -> FeatureId {
    FeatureId::new(family, ordinal).unwrap()
}

#[test]
fn ports_partial_shear_cost_gradient_mask_and_gradient_statistics() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();

    let mut registry = FeatureRegistry::new();
    register_basis_gradient(&mut registry).unwrap();
    let packet = registry
        .build(&[state], &policy(&[id(b'E', 1), id(b'E', 4), id(b'E', 5)]))
        .unwrap();

    let cost = packet.feature(id(b'E', 1));
    assert!(cost.row_is_available(0));
    assert_eq!(cost.values, vec![1.0]);

    let gradient = packet.feature(id(b'E', 4));
    assert!(gradient.row_is_available(0));
    let offset = |source: usize, target: usize, channel: usize| (source * 4 + target) * 2 + channel;
    assert_eq!(gradient.values[offset(0, 1, 0)], 2.0);
    assert_eq!(gradient.values[offset(0, 1, 1)], 1.0);
    assert_eq!(gradient.values[offset(1, 0, 0)], 3.0);
    assert_eq!(gradient.values[offset(1, 0, 1)], 1.0);
    assert_eq!(gradient.values[offset(0, 0, 1)], 0.0);
    assert_eq!(gradient.values[offset(2, 3, 1)], 0.0);

    let statistics = packet.feature(id(b'E', 5));
    assert!(statistics.row_is_available(0));
    assert_eq!(statistics.values, vec![2.0, 2.5, 3.0, 0.0]);
}

#[test]
fn basis_gradient_marks_empty_partial_block_inapplicable() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let identity = MatrixState::identity(4, alphabet.id()).unwrap();
    let mut registry = FeatureRegistry::new();
    register_basis_gradient(&mut registry).unwrap();
    let packet = registry
        .build(
            &[identity],
            &policy(&[id(b'E', 1), id(b'E', 4), id(b'E', 5)]),
        )
        .unwrap();
    for feature in [id(b'E', 1), id(b'E', 4), id(b'E', 5)] {
        assert_eq!(
            packet.feature(feature).status,
            vec![FeatureStatus::Inapplicable]
        );
    }
}
