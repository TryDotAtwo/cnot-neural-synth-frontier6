use cnot103_native::{
    register_paid_basis_features, CnotAlphabet, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureRole, FeatureStatus, MatrixState, PaidBasisBlockContext, PaidShearConfig,
};

#[test]
fn d09_through_d14_are_derived_from_one_verified_paid_shear_candidate() {
    let alphabet = CnotAlphabet::all_to_all_unit(10).unwrap();
    let block = vec![
        vec![0, 0, 1, 0],
        vec![1, 0, 1, 0],
        vec![1, 1, 1, 1],
        vec![1, 1, 1, 0],
        vec![0, 0, 1, 1],
        vec![1, 0, 1, 0],
    ];
    let source_wires = vec![0, 1, 2, 3];
    let target_wires = vec![4, 5, 6, 7, 8, 9];
    let mut rows = (0..10).map(|wire| 1u64 << wire).collect::<Vec<_>>();
    for (target_index, target_wire) in target_wires.iter().copied().enumerate() {
        for (source_index, source_wire) in source_wires.iter().copied().enumerate() {
            if block[target_index][source_index] != 0 {
                rows[target_wire] |= 1u64 << source_wire;
            }
        }
    }
    let state = MatrixState::from_packed_rows(10, alphabet.id(), rows).unwrap();
    let context = PaidBasisBlockContext::new(10, source_wires, target_wires).unwrap();
    let mut registry = FeatureRegistry::new();
    register_paid_basis_features(
        &mut registry,
        vec![context],
        PaidShearConfig::source_exact(),
    )
    .unwrap();

    let ids = [9, 10, 11, 12, 13, 14].map(|ordinal| FeatureId::new(b'D', ordinal).unwrap());
    let policy = FeaturePolicy {
        enabled_ids: ids.to_vec(),
        seed: 0,
        schema_id: "paid-basis-source-golden-v1".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: Some(6),
        max_bytes: Some(1 << 20),
    };
    let packet = registry.build(&[state], &policy).unwrap();

    for id in ids {
        let feature = packet.feature(id);
        assert_eq!(feature.status, vec![FeatureStatus::Computed], "{id}");
    }
    assert_eq!(packet.feature(ids[0]).spec.role, FeatureRole::Group);
    assert_eq!(packet.feature(ids[0]).spec.channels, 1);
    assert_eq!(packet.feature(ids[0]).values, vec![1.0]);
    assert_eq!(packet.feature(ids[1]).values, vec![0.0]);
    assert_eq!(packet.feature(ids[2]).values, vec![10.0, 1.0]);
    assert_eq!(packet.feature(ids[3]).values, vec![12.0, 2.0]);
    assert_eq!(
        packet.feature(ids[4]).values,
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 12.0]
    );

    let d09 = packet.feature(ids[0]);
    assert_eq!(d09.shape, vec![1, 1, 1]);
    assert_eq!(d09.token_valid.as_deref(), Some(&[1][..]));
    assert_eq!(d09.wire_roles, vec!["source_wire_set", "target_wire_set"]);
    let incidence = d09.incidence.as_ref().unwrap();
    assert_eq!(
        &incidence[0..10],
        &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        &incidence[10..20],
        &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    let d14 = packet.feature(ids[5]);
    assert_eq!(d14.spec.role, FeatureRole::Pair);
    assert_eq!(d14.spec.channels, 4);
    assert_eq!(d14.shape, vec![1, 10, 10, 4]);
    let pair = (2 * 10 + 4) * 4;
    assert_eq!(&d14.values[pair..pair + 4], &[1.0, 1.0, 0.0, 0.0]);
    let prep_pair = (2 * 10) * 4;
    assert_eq!(&d14.values[prep_pair..prep_pair + 4], &[0.0, 2.0, 0.0, 0.0]);
    let sidecar = &d14.provenance[0];
    assert!(sidecar
        .iter()
        .any(|item| item.starts_with("endpoint_sha256=")));
}

#[test]
fn paid_basis_features_fail_closed_without_context_or_explicit_solver_policy() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let state = MatrixState::identity(4, alphabet.id()).unwrap();
    let id = FeatureId::new(b'D', 9).unwrap();
    let mut registry = FeatureRegistry::new();
    register_paid_basis_features(
        &mut registry,
        vec![PaidBasisBlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap()],
        PaidShearConfig::source_exact(),
    )
    .unwrap();
    let mut policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "paid-basis-explicit-policy-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: Some(1),
        max_bytes: Some(1 << 20),
    };
    assert!(registry.selected_specs(&policy).is_err());

    policy.solver_assisted = true;
    let packet = registry.build(&[state], &policy).unwrap();
    assert_eq!(packet.feature(id).status, vec![FeatureStatus::Computed]);
}
