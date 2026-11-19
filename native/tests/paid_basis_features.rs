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
    // The canonical packet/model layout is [batch, group, node, role].
    // Check physical membership by wire, not a producer-specific flat layout.
    for wire in 0..10 {
        assert_eq!(incidence[wire * 2], f32::from(wire < 4));
        assert_eq!(incidence[wire * 2 + 1], f32::from(wire >= 4));
    }

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

#[test]
fn paid_basis_incidence_attaches_each_role_to_its_physical_wire_in_padded_batches() {
    let physical_contexts = [
        (5, vec![0, 3], vec![1, 4]),
        (7, vec![1, 5], vec![0, 6]),
        (7, vec![2, 6], vec![1, 4]),
    ];
    let contexts = physical_contexts
        .iter()
        .map(|(n, source, target)| {
            PaidBasisBlockContext::new(*n, source.clone(), target.clone()).unwrap()
        })
        .collect();
    let mut registry = FeatureRegistry::new();
    register_paid_basis_features(&mut registry, contexts, PaidShearConfig::source_exact()).unwrap();
    let ids = (9..=13)
        .map(|ordinal| FeatureId::new(b'D', ordinal).unwrap())
        .collect::<Vec<_>>();
    let policy = FeaturePolicy {
        enabled_ids: ids.clone(),
        seed: 0,
        schema_id: "paid-basis-physical-incidence-v2".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let dimensions = [5, 7, 3];
    let states = dimensions
        .iter()
        .map(|&n| {
            let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
            MatrixState::identity(n, alphabet.id()).unwrap()
        })
        .collect::<Vec<_>>();
    let packet = registry.build(&states, &policy).unwrap();
    for id in ids {
        let entry = packet.feature(id);
        assert_eq!(entry.shape[0..2], [3, 2]);
        assert_eq!(
            entry.status,
            [
                FeatureStatus::Computed,
                FeatureStatus::Computed,
                FeatureStatus::Unsupported
            ]
        );
        assert_eq!(entry.token_valid.as_deref(), Some(&[1, 0, 1, 1, 0, 0][..]));
        assert_eq!(entry.wire_roles, ["source_wire_set", "target_wire_set"]);
        let incidence = entry.incidence.as_ref().unwrap();
        assert_eq!(incidence.len(), 3 * 2 * 7 * 2);
        for (batch, n) in dimensions.iter().enumerate() {
            let matching = physical_contexts
                .iter()
                .filter(|c| c.0 == *n)
                .collect::<Vec<_>>();
            for group in 0..2 {
                for wire in 0..7 {
                    for role in 0..2 {
                        let expected = matching.get(group).is_some_and(|c| {
                            if role == 0 {
                                c.1.contains(&wire)
                            } else {
                                c.2.contains(&wire)
                            }
                        });
                        assert_eq!(
                            incidence[((batch * 2 + group) * 7 + wire) * 2 + role],
                            f32::from(expected),
                            "{id}, batch={batch}, group={group}, wire={wire}, role={role}"
                        );
                    }
                }
            }
        }
        assert!(entry
            .spec
            .producer_version
            .starts_with("verified_paid_basis_d09_d14_v2:"));
    }
}
