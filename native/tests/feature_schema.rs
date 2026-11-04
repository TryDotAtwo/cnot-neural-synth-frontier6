use cnot103_native::{
    all_feature_specs, register_general_compiler_bounds, schema::ProviderMetadata, CnotAlphabet,
    FeatureComputation, FeatureId, FeaturePolicy, FeatureProducer, FeatureRegistry, FeatureRole,
    FeatureSpec, FeatureValue, MatrixState,
};

struct DynamicPairFeature(FeatureSpec);

impl FeatureProducer for DynamicPairFeature {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let n = state.dimension() as usize;
        FeatureComputation::Computed(FeatureValue {
            shape: vec![n, n, 3],
            values: vec![0.0; n * n * 3],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

#[test]
fn d03_uses_the_source_ordered_pair_three_channel_layout() {
    let id = FeatureId::new(b'D', 3).unwrap();
    let spec = &all_feature_specs()[id.index()];

    assert_eq!(spec.id, id);
    assert_eq!(spec.role, FeatureRole::Pair);
    assert_eq!(spec.channels, 3);
}

#[test]
fn unimplemented_ids_are_reserved_until_a_provider_declares_their_shape() {
    for (family, ordinals) in [
        (b'C', vec![8]),
        (b'D', (9..=14).collect()),
        (b'E', vec![3, 6, 7, 8, 9]),
        (b'F', (3..=10).collect()),
        (b'H', (6..=7).collect()),
        (b'I', (1..=11).collect()),
    ] {
        for ordinal in ordinals {
            let id = FeatureId::new(family, ordinal).unwrap();
            let spec = &all_feature_specs()[id.index()];
            assert_eq!(spec.role, FeatureRole::Group, "{id}");
            assert_eq!(spec.channels, 1, "{id}");
            assert!(spec.wire_roles.is_empty(), "{id}");
            assert_eq!(spec.producer_version, "unconfigured", "{id}");
        }
    }
}

#[test]
fn provider_can_declare_its_actual_dynamic_shape() {
    let id = FeatureId::new(b'I', 11).unwrap();
    let mut spec = all_feature_specs()[id.index()].clone();
    spec.role = FeatureRole::Pair;
    spec.channels = 3;
    spec.wire_roles.clear();
    spec = spec.with_provider_metadata(
        ProviderMetadata::new("full_labeled_matrix", "fixture_i11_pair_v1", "exact", "T1")
            .dependencies(["M"]),
    );

    let mut registry = FeatureRegistry::new();
    registry
        .register(Box::new(DynamicPairFeature(spec)))
        .expect("the provider owns the actual layout for its feature");

    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let state = MatrixState::identity(2, alphabet.id()).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "dynamic-feature-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&[state], &policy).unwrap();
    let feature = packet.feature(id);
    assert_eq!(feature.spec.role, FeatureRole::Pair);
    assert_eq!(feature.spec.channels, 3);
    assert_eq!(feature.shape, vec![1, 2, 2, 3]);
    assert_eq!(feature.values, vec![0.0; 12]);
}

#[test]
fn selected_specs_follow_policy_and_reject_unimplemented_ids() {
    let id = FeatureId::new(b'I', 11).unwrap();
    let mut spec = all_feature_specs()[id.index()].clone();
    spec.role = FeatureRole::Pair;
    spec.channels = 3;
    spec.wire_roles.clear();
    spec = spec.with_provider_metadata(
        ProviderMetadata::new("full_labeled_matrix", "fixture_i11_pair_v1", "exact", "T1")
            .dependencies(["M"]),
    );

    let mut registry = FeatureRegistry::new();
    registry
        .register(Box::new(DynamicPairFeature(spec.clone())))
        .unwrap();
    let builtin_id = FeatureId::new(b'G', 5).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id, builtin_id],
        seed: 7,
        schema_id: "selected-specs-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };

    let selected = registry.selected_specs(&policy).unwrap();
    assert_eq!(
        selected.iter().map(|spec| spec.id).collect::<Vec<_>>(),
        vec![id, builtin_id]
    );
    assert_eq!(selected[0], spec);

    let mut missing_policy = policy;
    missing_policy.enabled_ids = vec![FeatureId::new(b'C', 8).unwrap()];
    assert!(registry.selected_specs(&missing_policy).is_err());
}

#[test]
fn selected_policy_reports_every_missing_provider_in_one_error() {
    let registry = FeatureRegistry::new();
    let expected_missing = [
        (b'C', 8),
        (b'D', 9),
        (b'D', 10),
        (b'D', 11),
        (b'D', 12),
        (b'D', 13),
        (b'D', 14),
        (b'E', 3),
        (b'E', 6),
        (b'E', 7),
        (b'E', 8),
        (b'E', 9),
        (b'F', 2),
        (b'F', 3),
        (b'F', 4),
        (b'F', 5),
        (b'F', 6),
        (b'F', 7),
        (b'F', 8),
        (b'F', 9),
        (b'F', 10),
        (b'H', 6),
        (b'H', 7),
        (b'I', 1),
        (b'I', 2),
        (b'I', 3),
        (b'I', 4),
        (b'I', 5),
        (b'I', 6),
        (b'I', 7),
        (b'I', 8),
        (b'I', 9),
        (b'I', 11),
    ];
    let mut policy = FeaturePolicy::complete_103("coverage-audit-v1");
    policy.enabled_ids = expected_missing
        .iter()
        .map(|(family, ordinal)| FeatureId::new(*family, *ordinal).unwrap())
        .collect();
    let error = registry.selected_specs(&policy).unwrap_err().to_string();
    for (family, ordinal) in expected_missing {
        let id = FeatureId::new(family, ordinal).unwrap().to_string();
        assert!(error.contains(&id), "coverage error omitted {id}: {error}");
    }
    assert_eq!(error.matches("no executable native provider").count(), 33);
}

#[test]
fn f09_f10_are_registered_as_explicit_global_solver_gated_contracts() {
    let mut registry = FeatureRegistry::new();
    register_general_compiler_bounds(&mut registry).unwrap();
    let f09 = FeatureId::new(b'F', 9).unwrap();
    let f10 = FeatureId::new(b'F', 10).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![f09, f10],
        seed: 0,
        schema_id: "f09-f10-contract-test-v1".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };

    let specs = registry.selected_specs(&policy).unwrap();
    assert_eq!(
        specs.iter().map(|spec| spec.id).collect::<Vec<_>>(),
        vec![f09, f10]
    );
    assert_eq!(specs[0].role, FeatureRole::Global);
    assert_eq!(specs[0].channels, 4);
    assert_eq!(specs[1].role, FeatureRole::Global);
    assert_eq!(specs[1].channels, 5);
    assert!(specs.iter().all(|spec| spec.requires_solver_assisted));
    assert!(specs.iter().all(|spec| spec
        .producer_version
        .contains("general_compiler_bounds_f09_f10_v1")));
}

#[test]
fn feature_registry_contract_hash_binds_provider_metadata_stably() {
    let id = FeatureId::new(b'I', 11).unwrap();
    let make_registry = |version: &str| {
        let mut registry = FeatureRegistry::new();
        let mut spec = all_feature_specs()[id.index()].clone();
        spec.role = FeatureRole::Pair;
        spec.channels = 3;
        spec.wire_roles.clear();
        let spec = spec.with_provider_metadata(
            ProviderMetadata::new("full_labeled_matrix", version, "exact", "T1")
                .dependencies(["M"]),
        );
        registry
            .register(Box::new(DynamicPairFeature(spec)))
            .unwrap();
        registry
    };
    let baseline = FeatureRegistry::new().contract_sha256();
    let first = make_registry("fixture_i11_pair_v1");
    let second = make_registry("fixture_i11_pair_v1");
    let changed = make_registry("fixture_i11_pair_v2");
    assert_eq!(first.contract_sha256().len(), 64);
    assert_eq!(first.contract_sha256(), second.contract_sha256());
    assert_ne!(first.contract_sha256(), baseline);
    assert_ne!(first.contract_sha256(), changed.contract_sha256());
}
