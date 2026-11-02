use cnot103_native::{
    register_guarded_pair_f02_view, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureRole, FeatureStatus, MatrixState,
};

fn policy(solver_assisted: bool) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'F', 2).unwrap()],
        seed: 0,
        schema_id: "f02-stage35-source-contract-v1".into(),
        solver_assisted,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

#[test]
fn f02_requires_the_explicit_solver_assisted_source_provider() {
    let registry = FeatureRegistry::new();
    assert!(registry.selected_specs(&policy(true)).is_err());
}

#[test]
fn f02_matches_the_source_only_for_exact_stage35_pair_certificates() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let identity = MatrixState::identity(4, alphabet.id()).unwrap();
    let mut one_edge = identity.clone();
    one_edge
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let mut swap = identity.clone();
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
        swap.apply_cnot(gate).unwrap();
    }

    let mut registry = FeatureRegistry::new();
    register_guarded_pair_f02_view(&mut registry, 4, 100_000).unwrap();
    cnot103_native::register_guarded_pair(&mut registry, 4, 100_000).unwrap();
    let packet = registry
        .build(&[identity, one_edge, swap], &policy(true))
        .unwrap();
    let feature = packet.feature(FeatureId::new(b'F', 2).unwrap());
    assert_eq!(feature.spec.role, FeatureRole::Global);
    assert_eq!(feature.spec.channels, 1);
    assert!(feature.spec.wire_roles.is_empty());
    assert!(feature.spec.requires_solver_assisted);
    assert_eq!(feature.shape, vec![3, 1]);
    assert_eq!(feature.status, vec![FeatureStatus::Computed; 3]);
    assert_eq!(feature.values, vec![0.0, 1.0, 3.0]);
    let specs = registry.specs();
    let f06 = specs
        .iter()
        .find(|spec| spec.id == FeatureId::new(b'F', 6).unwrap())
        .unwrap();
    assert_eq!(feature.spec.producer_version, f06.producer_version);
    assert_eq!(
        feature.spec.scope,
        "solver_assisted;stage35_verified_pair_scope_only"
    );
}

#[test]
fn f02_fails_closed_on_solver_gate_width_alphabet_and_theorem_scope() {
    let mut registry = FeatureRegistry::new();
    register_guarded_pair_f02_view(&mut registry, 3, 100_000).unwrap();
    assert!(registry.selected_specs(&policy(false)).is_err());

    let wide = MatrixState::identity(4, "all_to_all_unit_cnot:4").unwrap();
    let limited = registry.build(&[wide], &policy(true)).unwrap();
    assert_eq!(
        limited.feature(FeatureId::new(b'F', 2).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let outside_pair_scope =
        MatrixState::from_packed_rows(3, alphabet.id(), vec![2, 4, 1]).unwrap();
    let unsupported = registry
        .build(&[outside_pair_scope], &policy(true))
        .unwrap();
    assert_eq!(
        unsupported.feature(FeatureId::new(b'F', 2).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );

    let incomplete = CnotAlphabet::new(
        2,
        "incomplete-f02",
        vec![CnotGate {
            control: 0,
            target: 1,
        }],
        vec![1],
    )
    .unwrap();
    let state = MatrixState::identity(2, incomplete.id()).unwrap();
    let unsupported_alphabet = registry.build(&[state], &policy(true)).unwrap();
    assert_eq!(
        unsupported_alphabet
            .feature(FeatureId::new(b'F', 2).unwrap())
            .status,
        vec![FeatureStatus::Unsupported]
    );
}
