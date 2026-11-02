use cnot103_native::{
    register_matrix_boundaries, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole,
    FeatureStatus, MatrixBoundarySpec, MatrixState, RationalWeight,
};

fn state() -> MatrixState {
    MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", vec![3, 2, 4]).unwrap()
}

fn policy(ids: &[&str]) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: ids
            .iter()
            .map(|value| FeatureId::new(value.as_bytes()[0], value[1..].parse().unwrap()).unwrap())
            .collect(),
        seed: 0,
        schema_id: "matrix-boundary-source-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}

#[test]
fn source_boundary_fixture_preserves_rank_projection_order_and_incidence() {
    // Golden values are from research_v3.features.matrix_boundary and
    // tests/test_matrix_boundary.py for M rows (3,2,4), inputs (0,1),
    // output (0), protected inputs (0,1), and pinned zero output (0).
    let boundary = MatrixBoundarySpec::new(3, vec![0, 1], vec![0])
        .unwrap()
        .with_protected_inputs(vec![0, 1])
        .unwrap()
        .with_zero_outputs(vec![0])
        .unwrap();
    let mut registry = FeatureRegistry::new();
    register_matrix_boundaries(&mut registry, vec![boundary], 512, 512).unwrap();
    let packet = registry
        .build(
            &[state()],
            &policy(&["C08", "I01", "I02", "I03", "I04", "I11"]),
        )
        .unwrap();

    let c08 = packet.feature(FeatureId::new(b'C', 8).unwrap());
    assert_eq!(c08.spec.role, FeatureRole::Global);
    assert_eq!(c08.status, vec![FeatureStatus::Computed]);
    assert_eq!(c08.values, vec![1.0]);

    let i03 = packet.feature(FeatureId::new(b'I', 3).unwrap());
    assert_eq!(i03.values, vec![1.0, 1.0]);

    let i11 = packet.feature(FeatureId::new(b'I', 11).unwrap());
    assert_eq!(i11.values, vec![3.0, 1.0, 1.0, 0.0]);
    assert_eq!(
        packet.feature(FeatureId::new(b'I', 4).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );

    let i01 = packet.feature(FeatureId::new(b'I', 1).unwrap());
    assert_eq!(i01.spec.role, FeatureRole::Group);
    assert_eq!(i01.shape, vec![1, 3, 2]);
    assert_eq!(i01.token_valid, Some(vec![1, 1, 1]));
    let incidence = i01.incidence.as_ref().unwrap();
    let at = |group: usize, wire: usize, role: usize| incidence[(group * 3 + wire) * 3 + role];
    assert_eq!(at(0, 0, 0), 1.0);
    assert_eq!(at(0, 0, 2), 1.0);
    assert_eq!(at(1, 1, 0), 1.0);
    assert_eq!(at(1, 1, 2), 1.0);
    assert_eq!(at(2, 0, 1), 1.0);
    assert_eq!(at(2, 0, 2), 0.0);

    // Canonical equations are in descending-pivot order, matching the source
    // AffineRelation.canonical_equations() contract.
    let i02 = packet.feature(FeatureId::new(b'I', 2).unwrap());
    assert_eq!(i02.shape, vec![1, 2, 3]);
    assert_eq!(i02.values, vec![2.0, 1.0, 0.0, 2.0, 1.0, 0.0]);
    let equation_incidence = i02.incidence.as_ref().unwrap();
    let at = |equation: usize, wire: usize, role: usize| {
        equation_incidence[(equation * 3 + wire) * 3 + role]
    };
    assert_eq!(at(0, 0, 1), 1.0);
    assert_eq!(at(0, 0, 0), 0.0);
    assert_eq!(at(1, 0, 0), 1.0);
    assert_eq!(at(1, 1, 0), 1.0);
    assert_eq!(at(1, 0, 1), 0.0);
}

#[test]
fn i04_matches_the_source_zero_phase_weighted_boundary_descriptor() {
    let boundary = MatrixBoundarySpec::new(3, vec![0, 1], vec![0])
        .unwrap()
        .with_weights(vec![(
            RationalWeight::new(2, 1).unwrap(),
            RationalWeight::new(3, 1).unwrap(),
        )])
        .unwrap();
    let mut registry = FeatureRegistry::new();
    register_matrix_boundaries(&mut registry, vec![boundary], 512, 512).unwrap();
    let packet = registry.build(&[state()], &policy(&["I04"])).unwrap();
    let feature = packet.feature(FeatureId::new(b'I', 4).unwrap());
    assert_eq!(feature.spec.channels, 3);
    assert_eq!(feature.values, vec![1.0, 1.0, 1.0]);
}

#[test]
fn matrix_boundary_fail_closed_for_missing_context_policy_and_budgets() {
    let boundary = MatrixBoundarySpec::new(3, vec![0, 1], vec![0]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_matrix_boundaries(&mut registry, vec![boundary.clone()], 2, 512).unwrap();
    let packet = registry.build(&[state()], &policy(&["I02"])).unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'I', 2).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let mut byte_limited = FeaturePolicy {
        max_bytes: Some(1),
        ..policy(&["C08"])
    };
    byte_limited.symmetry_contract = "augmentation_only".into();
    let mut byte_registry = FeatureRegistry::new();
    register_matrix_boundaries(
        &mut byte_registry,
        vec![MatrixBoundarySpec::new(3, vec![0, 1], vec![0]).unwrap()],
        512,
        512,
    )
    .unwrap();
    let byte_packet = byte_registry.build(&[state()], &byte_limited).unwrap();
    assert_eq!(
        byte_packet.feature(FeatureId::new(b'C', 8).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let mut unregistered = FeatureRegistry::new();
    register_matrix_boundaries(&mut unregistered, vec![boundary], 512, 512).unwrap();
    let mut exact = policy(&["C08"]);
    exact.symmetry_contract = "exact_relabel".into();
    assert!(unregistered.selected_specs(&exact).is_err());

    let no_width = MatrixState::identity(4, "all_to_all_unit_cnot:4").unwrap();
    let unsupported = unregistered.build(&[no_width], &policy(&["C08"])).unwrap();
    assert_eq!(
        unsupported.feature(FeatureId::new(b'C', 8).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );
}

#[test]
fn matrix_boundary_spec_rejects_noncanonical_or_inconsistent_physical_wires() {
    assert!(MatrixBoundarySpec::new(3, vec![1, 0], vec![2]).is_err());
    assert!(MatrixBoundarySpec::new(3, vec![], vec![2]).is_err());
    assert!(MatrixBoundarySpec::new(3, vec![0], vec![3]).is_err());
    assert!(MatrixBoundarySpec::new(3, vec![0], vec![1])
        .unwrap()
        .with_protected_inputs(vec![1])
        .is_err());
    assert!(RationalWeight::new(1, 0).is_err());
    assert!(RationalWeight::new("-1", "3").is_err());
    assert!(RationalWeight::new("340282366920938463463374607431768211456", "3").is_ok());
    assert!(register_matrix_boundaries(
        &mut FeatureRegistry::new(),
        vec![
            MatrixBoundarySpec::new(3, vec![0], vec![1]).unwrap(),
            MatrixBoundarySpec::new(3, vec![0], vec![2]).unwrap(),
        ],
        512,
        512,
    )
    .is_err());
}
