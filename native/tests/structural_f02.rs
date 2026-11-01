use cnot103_native::{
    CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, MatrixState,
};

fn f02(state: &MatrixState) -> f32 {
    let id = FeatureId::new(b'F', 2).unwrap();
    let registry = FeatureRegistry::new();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "f02-connectivity-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&[state.clone()], &policy).unwrap();
    let feature = packet.feature(id);
    assert!(
        feature.row_is_available(0),
        "F02 must be computed from the matrix"
    );
    feature.values[0]
}

#[test]
fn f02_is_connectivity_plus_zero_diagonal_count() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();

    let identity = MatrixState::identity(4, alphabet.id()).unwrap();
    assert_eq!(f02(&identity), 0.0);

    let mut one_component_edge = identity.clone();
    one_component_edge
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    assert_eq!(f02(&one_component_edge), 1.0);

    let mut two_zero_diagonal_entries = identity;
    for gate in [
        CnotGate {
            control: 0,
            target: 1,
        },
        CnotGate {
            control: 1,
            target: 0,
        },
        CnotGate {
            control: 0,
            target: 1,
        },
    ] {
        two_zero_diagonal_entries.apply_cnot(gate).unwrap();
    }
    assert_eq!(f02(&two_zero_diagonal_entries), 3.0);
}
