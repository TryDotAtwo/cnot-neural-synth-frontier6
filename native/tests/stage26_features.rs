use cnot103_native::{
    register_stage26_features, register_stage26_features_with_budget, CnotAlphabet, CnotGate,
    FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState,
    STAGE26_CERTIFICATE_SOURCE_SHA256, STAGE26_CHECKER_SOURCE_SHA256,
};
use std::collections::{HashMap, VecDeque};

#[derive(serde::Deserialize)]
struct Stage26Golden {
    schema: String,
    source_sha256: String,
    checker_sha256: String,
    cases: Vec<Stage26GoldenCase>,
}

#[derive(serde::Deserialize)]
struct Stage26GoldenCase {
    rows: Vec<u64>,
    distance: usize,
    f03: Vec<f32>,
    f04: Vec<f32>,
    f05: Vec<f32>,
}

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: [3, 4, 5]
            .into_iter()
            .map(|ordinal| FeatureId::new(b'F', ordinal).unwrap())
            .collect(),
        seed: 0,
        schema_id: "stage26-native-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn gl3_distances() -> HashMap<Vec<u64>, usize> {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let identity = MatrixState::identity(3, alphabet.id()).unwrap();
    let mut distances = HashMap::from([(identity.packed_rows().to_vec(), 0usize)]);
    let mut queue = VecDeque::from([identity]);
    while let Some(state) = queue.pop_front() {
        let distance = distances[state.packed_rows()];
        for control in 0..3 {
            for target in 0..3 {
                if control == target {
                    continue;
                }
                let mut next = state.clone();
                next.apply_cnot(CnotGate { control, target }).unwrap();
                if !distances.contains_key(next.packed_rows()) {
                    distances.insert(next.packed_rows().to_vec(), distance + 1);
                    queue.push_back(next);
                }
            }
        }
    }
    distances
}

#[test]
fn stage26_features_match_source_golden_and_never_overstate_gl3_distance() {
    let mut registry = FeatureRegistry::new();
    register_stage26_features(&mut registry).unwrap();
    let policy = policy();
    let golden: Stage26Golden =
        serde_json::from_str(include_str!("fixtures/stage26_gl3_source_golden.json")).unwrap();
    assert_eq!(golden.schema, "stage26_gl3_source_golden_v1");
    assert_eq!(golden.source_sha256, STAGE26_CERTIFICATE_SOURCE_SHA256);
    assert_eq!(golden.checker_sha256, STAGE26_CHECKER_SOURCE_SHA256);
    let source_cases: HashMap<_, _> = golden
        .cases
        .into_iter()
        .map(|case| (case.rows.clone(), case))
        .collect();
    assert_eq!(source_cases.len(), 168, "source fixture must cover GL(3,2)");

    for (rows, distance) in gl3_distances() {
        let source = source_cases
            .get(&rows)
            .expect("source fixture covers every GL3 state");
        assert_eq!(
            source.distance, distance,
            "source distance mismatch at {rows:?}"
        );
        let state =
            MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", rows.clone()).unwrap();
        let packet = registry.build(&[state], &policy).unwrap();
        assert_eq!(
            packet.feature(FeatureId::new(b'F', 3).unwrap()).values,
            source.f03,
            "F03 differs from source for {rows:?}"
        );
        assert_eq!(
            packet.feature(FeatureId::new(b'F', 4).unwrap()).values,
            source.f04,
            "F04 differs from source for {rows:?}"
        );
        let f05 = packet.feature(FeatureId::new(b'F', 5).unwrap());
        assert_eq!(f05.status, vec![FeatureStatus::Computed]);
        assert_eq!(
            f05.values, source.f05,
            "F05 differs from source for {rows:?}"
        );
        assert!(
            f05.values[0] as usize <= distance,
            "F05 exceeded exact GL(3,2) distance"
        );
    }
}

#[test]
fn stage26_fails_closed_for_alphabet_and_budget_mismatches() {
    let mut registry = FeatureRegistry::new();
    register_stage26_features(&mut registry).unwrap();
    let state = MatrixState::identity(3, "restricted-cnot:3").unwrap();
    let packet = registry.build(&[state], &policy()).unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'F', 3).unwrap()).status,
        vec![FeatureStatus::Unsupported]
    );

    let mut bounded = FeatureRegistry::new();
    register_stage26_features_with_budget(&mut bounded, 2, 100).unwrap();
    let state = MatrixState::identity(3, "all_to_all_unit_cnot:3").unwrap();
    let packet = bounded.build(&[state], &policy()).unwrap();
    assert_eq!(
        packet.feature(FeatureId::new(b'F', 5).unwrap()).status,
        vec![FeatureStatus::ResourceLimit]
    );
}
