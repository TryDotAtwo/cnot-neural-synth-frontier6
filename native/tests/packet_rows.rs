use cnot103_native::{
    all_feature_specs, CnotAlphabet, CnotGate, FeatureBuildError, FeatureComputation, FeatureId,
    FeaturePolicy, FeatureProducer, FeatureRegistry, FeatureSpec, FeatureValue, MatrixState,
};

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
    assert!(!packet.complete_103());

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
