use cnot103_native::{
    register_guarded_one_excess, register_guarded_pair, register_guarded_zero_excess, CnotAlphabet,
    CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole, FeatureStatus, MatrixState,
};

fn policy(solver_assisted: bool) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'F', 7).unwrap()],
        seed: 0,
        schema_id: "f07-source-guarded-zero-excess-v1".into(),
        solver_assisted,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn one_excess_policy(solver_assisted: bool) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'F', 8).unwrap()],
        seed: 0,
        schema_id: "f08-source-guarded-one-excess-v1".into(),
        solver_assisted,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn pair_policy(solver_assisted: bool) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'F', 6).unwrap()],
        seed: 0,
        schema_id: "f06-source-guarded-pair-v1".into(),
        solver_assisted,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}

#[test]
fn f07_matches_the_source_three_status_feature_without_widening_its_claim() {
    let swap_alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let swap = MatrixState::from_packed_rows(2, swap_alphabet.id(), vec![2, 1]).unwrap();
    let identity = MatrixState::identity(2, swap_alphabet.id()).unwrap();
    let mut registry = FeatureRegistry::new();
    register_guarded_zero_excess(&mut registry, 3).unwrap();
    let packet = registry.build(&[swap, identity], &policy(true)).unwrap();
    let feature = packet.feature(FeatureId::new(b'F', 7).unwrap());
    assert_eq!(feature.spec.role, FeatureRole::Global);
    assert_eq!(feature.spec.channels, 3);
    assert_eq!(feature.shape, vec![2, 3]);
    assert_eq!(
        feature.status,
        vec![FeatureStatus::Computed, FeatureStatus::Computed]
    );
    assert_eq!(feature.values, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn f07_fails_closed_on_solver_gate_width_and_alphabet() {
    let mut registry = FeatureRegistry::new();
    register_guarded_zero_excess(&mut registry, 2).unwrap();
    assert!(registry.selected_specs(&policy(false)).is_err());

    let wide = MatrixState::identity(3, "all_to_all_unit_cnot:3").unwrap();
    let limited = registry.build(&[wide], &policy(true)).unwrap();
    assert_eq!(
        limited.feature(FeatureId::new(b'F', 7).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let incomplete = CnotAlphabet::new(
        2,
        "incomplete-f07",
        vec![CnotGate {
            control: 0,
            target: 1,
        }],
        vec![1],
    )
    .unwrap();
    let state = MatrixState::identity(2, incomplete.id()).unwrap();
    let unsupported = registry.build(&[state], &policy(true)).unwrap();
    assert_eq!(
        unsupported.feature(FeatureId::new(b'F', 7).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );
}

#[test]
fn f07_producer_identity_matches_the_python_reference_bundle() {
    let mut registry = FeatureRegistry::new();
    register_guarded_zero_excess(&mut registry, 3).unwrap();
    let spec = registry
        .specs()
        .into_iter()
        .find(|spec| spec.id == FeatureId::new(b'F', 7).unwrap())
        .unwrap();
    assert_eq!(
        spec.producer_version,
        "guarded_certificate_features_v1:5ebfbcf8f9e79b1b0aa7b3154e3a4663772a7d5f51f355745c1f0e5d94f2a2ba"
    );
    assert_eq!(spec.role, FeatureRole::Global);
    assert_eq!(spec.channels, 3);
    assert!(spec.wire_roles.is_empty());
}

#[test]
fn f08_matches_source_exact_negative_unsupported_statuses() {
    let alphabet2 = CnotAlphabet::all_to_all_unit(2).unwrap();
    let alphabet3 = CnotAlphabet::all_to_all_unit(3).unwrap();
    let swap = MatrixState::from_packed_rows(2, alphabet2.id(), vec![2, 1]).unwrap();
    let exact = MatrixState::from_packed_rows(3, alphabet3.id(), vec![1, 7, 4]).unwrap();
    let negative = MatrixState::from_packed_rows(3, alphabet3.id(), vec![6, 1, 4]).unwrap();
    let identity = MatrixState::identity(3, alphabet3.id()).unwrap();
    let mut registry = FeatureRegistry::new();
    register_guarded_one_excess(&mut registry, 3, 100_000).unwrap();
    let packet = registry
        .build(&[swap, exact, negative, identity], &one_excess_policy(true))
        .unwrap();
    let feature = packet.feature(FeatureId::new(b'F', 8).unwrap());
    assert_eq!(feature.spec.role, FeatureRole::Global);
    assert_eq!(feature.spec.channels, 3);
    assert_eq!(feature.shape, vec![4, 3]);
    assert_eq!(feature.status, vec![FeatureStatus::Computed; 4]);
    assert_eq!(
        feature.values,
        vec![
            1.0, 0.0, 0.0, // two-wire swap, exact one-excess word
            1.0, 0.0, 0.0, // source GL(3,2) exact sample
            0.0, 1.0, 0.0, // only lengths r and r+1 are excluded
            0.0, 0.0, 1.0, // zero-excess labels belong to F07
        ]
    );
}

#[test]
fn f08_budget_and_solver_gate_never_become_negative_labels() {
    let alphabet2 = CnotAlphabet::all_to_all_unit(2).unwrap();
    let swap = MatrixState::from_packed_rows(2, alphabet2.id(), vec![2, 1]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_guarded_one_excess(&mut registry, 2, 1).unwrap();
    assert!(registry.selected_specs(&one_excess_policy(false)).is_err());
    let limited = registry.build(&[swap], &one_excess_policy(true)).unwrap();
    let feature = limited.feature(FeatureId::new(b'F', 8).unwrap());
    assert_eq!(feature.status, vec![FeatureStatus::ResourceLimit]);
    assert_eq!(feature.available, vec![0]);
}

#[test]
fn f06_matches_source_six_channels_and_physical_wire_incidence() {
    let alphabet2 = CnotAlphabet::all_to_all_unit(2).unwrap();
    let alphabet3 = CnotAlphabet::all_to_all_unit(3).unwrap();
    let swap = MatrixState::from_packed_rows(2, alphabet2.id(), vec![2, 1]).unwrap();
    let identity = MatrixState::identity(3, alphabet3.id()).unwrap();
    let mut registry = FeatureRegistry::new();
    register_guarded_pair(&mut registry, 3, 100_000).unwrap();
    register_guarded_zero_excess(&mut registry, 3).unwrap();
    let packet = registry
        .build(&[swap, identity], &pair_policy(true))
        .unwrap();
    let feature = packet.feature(FeatureId::new(b'F', 6).unwrap());
    assert_eq!(feature.spec.role, FeatureRole::Group);
    assert_eq!(feature.spec.channels, 6);
    assert_eq!(
        feature.wire_roles,
        vec!["active_first", "active_second", "external_sources"]
    );
    assert_eq!(feature.shape, vec![2, 1, 6]);
    assert_eq!(feature.status, vec![FeatureStatus::Computed; 2]);
    assert_eq!(feature.available, vec![1, 1]);
    assert_eq!(
        feature.values,
        vec![
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // stage-35 frame for the two-wire swap
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // source identity frame
        ]
    );
    assert_eq!(feature.token_valid, Some(vec![1, 1]));
    assert_eq!(
        feature.incidence,
        Some(vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // padded two-wire state
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // third wire is external
        ])
    );
    let specs = registry.specs();
    let f06 = specs
        .iter()
        .find(|spec| spec.id == FeatureId::new(b'F', 6).unwrap())
        .unwrap();
    let f07 = specs
        .iter()
        .find(|spec| spec.id == FeatureId::new(b'F', 7).unwrap())
        .unwrap();
    assert_eq!(f06.producer_version, f07.producer_version);
    assert_eq!(
        f06.scope,
        "solver_assisted;stage35_verified_pair_scope_only"
    );
    assert_eq!(
        f06.dependencies,
        vec![
            "M",
            "full_unit_alphabet",
            "theorem_sources",
            "explicit_oracle_budget"
        ]
    );
}

#[test]
fn f06_is_fail_closed_outside_its_exact_pair_scope() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let three_changed_rows =
        MatrixState::from_packed_rows(3, alphabet.id(), vec![2, 4, 1]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_guarded_pair(&mut registry, 3, 100_000).unwrap();

    let mut raw_policy = pair_policy(false);
    assert!(registry.selected_specs(&raw_policy).is_err());
    raw_policy.solver_assisted = true;
    raw_policy.symmetry_contract = "exact_relabel".into();
    assert!(registry.selected_specs(&raw_policy).is_err());

    let unsupported = registry
        .build(&[three_changed_rows.clone()], &pair_policy(true))
        .unwrap();
    let feature = unsupported.feature(FeatureId::new(b'F', 6).unwrap());
    assert_eq!(feature.status, vec![FeatureStatus::Unsupported]);
    assert_eq!(feature.available, vec![0]);

    let mut limited_registry = FeatureRegistry::new();
    register_guarded_pair(&mut limited_registry, 2, 100_000).unwrap();
    let limited = limited_registry
        .build(&[three_changed_rows], &pair_policy(true))
        .unwrap();
    assert_eq!(
        limited.feature(FeatureId::new(b'F', 6).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let incomplete = CnotAlphabet::new(
        2,
        "incomplete-f06",
        vec![CnotGate {
            control: 0,
            target: 1,
        }],
        vec![1],
    )
    .unwrap();
    let incomplete_state = MatrixState::identity(2, incomplete.id()).unwrap();
    let unsupported_alphabet = limited_registry
        .build(&[incomplete_state], &pair_policy(true))
        .unwrap();
    assert_eq!(
        unsupported_alphabet
            .feature(FeatureId::new(b'F', 6).unwrap())
            .status,
        vec![FeatureStatus::Unsupported]
    );
}
