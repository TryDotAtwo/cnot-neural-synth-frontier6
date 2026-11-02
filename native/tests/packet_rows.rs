use cnot103_native::{
    all_feature_specs, CnotAlphabet, CnotGate, FeatureBuildError, FeatureComputation, FeatureId,
    FeaturePolicy, FeatureProducer, FeatureRegistry, FeatureRole, FeatureSpec, FeatureStatus,
    FeatureValue, MatrixState,
};

struct AllInputsFixture {
    spec: FeatureSpec,
}

impl FeatureProducer for AllInputsFixture {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        if self.spec.id == FeatureId::new(b'C', 5).unwrap() && state.is_identity() {
            return FeatureComputation::Unavailable {
                status: FeatureStatus::Inapplicable,
                reason: "fixture_feature_is_inapplicable_at_identity".into(),
            };
        }

        let n = state.dimension() as usize;
        let (shape, count) = match self.spec.role {
            FeatureRole::Node => (vec![n, self.spec.channels], n * self.spec.channels),
            FeatureRole::Pair => (vec![n, n, self.spec.channels], n * n * self.spec.channels),
            FeatureRole::Global => (vec![self.spec.channels], self.spec.channels),
            FeatureRole::Group => (vec![1, self.spec.channels], self.spec.channels),
        };
        FeatureComputation::Computed(FeatureValue {
            shape,
            values: vec![0.0; count],
            token_valid: if self.spec.role == FeatureRole::Group {
                vec![1]
            } else {
                Vec::new()
            },
            incidence: if self.spec.role == FeatureRole::Group {
                vec![0.0; n * self.spec.wire_roles.len()]
            } else {
                Vec::new()
            },
        })
    }
}

fn all_model_input_registry() -> (FeatureRegistry, FeaturePolicy) {
    let mut registry = FeatureRegistry::new();
    let mut policy = FeaturePolicy::all_model_inputs_augmentation_only("packet-all-inputs-test-v1");
    policy.solver_assisted = true;

    for spec in all_feature_specs() {
        let id = spec.id;
        if id == FeatureId::new(b'H', 7).unwrap() {
            continue;
        }
        let mut one_id_policy = policy.clone();
        one_id_policy.enabled_ids = vec![id];
        if registry.selected_specs(&one_id_policy).is_ok() {
            continue;
        }

        let mut spec = registry
            .specs()
            .into_iter()
            .find(|candidate| candidate.id == id)
            .unwrap();
        let wire_roles = if spec.role == FeatureRole::Group && spec.wire_roles.is_empty() {
            vec!["source".to_owned(), "target".to_owned()]
        } else {
            spec.wire_roles.clone()
        };
        spec = spec.with_provider_metadata(
            "packet_rows_synthetic_all_inputs",
            "packet_rows_synthetic_all_inputs_v1",
            ["M"],
            "exact_fixture",
            "T1",
            wire_roles,
            false,
        );
        registry
            .register(Box::new(AllInputsFixture { spec }))
            .unwrap();
    }
    (registry, policy)
}

struct GroupFixture {
    spec: FeatureSpec,
}

impl FeatureProducer for GroupFixture {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let ones = state
            .packed_rows()
            .iter()
            .map(|row| row.count_ones())
            .sum::<u32>() as f32;
        FeatureComputation::Computed(FeatureValue {
            shape: vec![1, 1],
            values: vec![ones],
            token_valid: vec![1],
            incidence: vec![ones; state.dimension() as usize * 2],
        })
    }
}

fn registry() -> Result<(FeatureRegistry, FeatureId), FeatureBuildError> {
    let id = FeatureId::new(b'C', 2).unwrap();
    let mut spec = all_feature_specs()
        .into_iter()
        .find(|spec| spec.id == id)
        .unwrap();
    spec.scope = "fixture_group".into();
    spec.producer_version = "fixture_group_v1".into();
    spec.dependencies = vec!["M".into()];
    spec.exactness = "exact".into();
    spec.cost_tier = "T1".into();
    spec.wire_roles = vec!["source".into(), "target".into()];

    let mut registry = FeatureRegistry::new();
    registry.register(Box::new(GroupFixture { spec }))?;
    Ok((registry, id))
}

#[test]
fn model_input_coverage_ignores_h07_and_allows_inapplicability() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, alphabet.id()).unwrap();
    let (registry, policy) = all_model_input_registry();
    let packet = registry.build(&[state], &policy).unwrap();

    assert_eq!(
        packet.feature(FeatureId::new(b'H', 7).unwrap()).status,
        vec![FeatureStatus::NotRequested]
    );
    assert_eq!(
        packet.feature(FeatureId::new(b'C', 5).unwrap()).status,
        vec![FeatureStatus::Inapplicable]
    );
    assert!(
        packet.require_complete_model_input_coverage().is_ok(),
        "H07 is not a model input and an inapplicable feature is represented by its mask"
    );
}

#[test]
fn single_row_preserves_feature_values_masks_group_metadata_and_identity() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let identity = MatrixState::identity(3, alphabet.id()).unwrap();
    let mut moved = identity.clone();
    moved
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let states = [identity, moved];
    let (registry, group_id) = registry().unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![group_id],
        seed: 7,
        schema_id: "feature-fixture-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&states, &policy).unwrap();
    assert!(!packet.has_complete_model_input_coverage());
    assert!(packet.require_complete_model_input_coverage().is_err());

    let selected = packet.single_row(1).unwrap();
    assert_eq!(selected.batch, 1);
    assert_eq!(selected.state_keys, vec![packet.state_keys[1].clone()]);
    assert_eq!(selected.alphabet_ids, vec![alphabet.id().to_owned()]);
    assert_eq!(selected.valid_nodes, vec![1, 1, 1]);
    assert_eq!(selected.schema_id, packet.schema_id);
    assert_eq!(selected.policy_id, packet.policy_id);

    let group = selected.feature(group_id);
    assert_eq!(group.shape, vec![1, 1, 1]);
    assert_eq!(group.values, vec![4.0]);
    assert_eq!(group.available, vec![1]);
    assert_eq!(group.token_valid, Some(vec![1]));
    assert_eq!(group.incidence, Some(vec![4.0; 6]));

    let unrequested = selected.feature(FeatureId::new(b'A', 1).unwrap());
    assert_eq!(unrequested.available, vec![0]);
    assert_eq!(
        unrequested.status,
        vec![cnot103_native::FeatureStatus::NotRequested]
    );
}
