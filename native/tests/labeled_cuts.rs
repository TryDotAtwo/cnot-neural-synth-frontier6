use cnot103_native::schema::feature_spec;
use cnot103_native::{
    register_labeled_cuts, register_rank_increment_probes, register_restricted_intersections,
    FeatureComputation, FeatureId, FeaturePolicy, FeatureProducer, FeatureRegistry, FeatureStatus,
    LabeledCut, MatrixState, RankIncrementProbe, RestrictedIntersection,
};

struct ExistingC03(cnot103_native::FeatureSpec);

impl FeatureProducer for ExistingC03 {
    fn spec(&self) -> &cnot103_native::FeatureSpec {
        &self.0
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, _state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        FeatureComputation::Unavailable {
            status: FeatureStatus::Unsupported,
            reason: "test_provider".into(),
        }
    }
}

fn cut_policy(symmetry_contract: &str) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: (2..=4)
            .map(|ordinal| FeatureId::new(b'C', ordinal).unwrap())
            .collect(),
        seed: 0,
        schema_id: "cut-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes: None,
    }
}

#[test]
fn labeled_cut_features_compute_three_exact_block_ranks_and_incidence() {
    // M = [[1,1,0],[0,1,1],[0,0,1]], A={0,1}, B={2}.
    // The three ranks are rank(M[A,B])=1, rank(M[B,A])=0,
    // and rank((M-I)[A,A])=1 over GF(2).
    let state =
        MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", vec![0b011, 0b110, 0b100])
            .unwrap();
    let cut = LabeledCut::new(3, vec![0, 1]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_labeled_cuts(&mut registry, vec![cut]).unwrap();

    let packet = registry
        .build(&[state], &cut_policy("augmentation_only"))
        .unwrap();

    for (ordinal, expected) in [(2, 1.0), (3, 0.0), (4, 1.0)] {
        let feature = packet.feature(FeatureId::new(b'C', ordinal).unwrap());
        assert!(feature.row_is_available(0));
        assert_eq!(feature.values, vec![expected]);
        assert_eq!(feature.token_valid, Some(vec![1]));
        assert_eq!(feature.incidence, Some(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0]));
        assert_eq!(feature.wire_roles, vec!["cut_A", "cut_B"]);
    }
}

#[test]
fn labeled_cut_provider_rejects_exact_relabel_policy() {
    let state = MatrixState::identity(3, "all_to_all_unit_cnot:3").unwrap();
    let mut registry = FeatureRegistry::new();
    register_labeled_cuts(&mut registry, vec![LabeledCut::new(3, vec![0]).unwrap()]).unwrap();

    let error = registry
        .build(&[state], &cut_policy("exact_relabel"))
        .unwrap_err();

    assert!(error.0.contains("provider symmetry contract mismatch"));
}

#[test]
fn labeled_cut_provider_marks_a_different_matrix_width_unsupported() {
    let state = MatrixState::identity(2, "all_to_all_unit_cnot:2").unwrap();
    let mut registry = FeatureRegistry::new();
    register_labeled_cuts(&mut registry, vec![LabeledCut::new(3, vec![0]).unwrap()]).unwrap();

    let packet = registry
        .build(&[state], &cut_policy("augmentation_only"))
        .unwrap();

    for ordinal in 2..=4 {
        let feature = packet.feature(FeatureId::new(b'C', ordinal).unwrap());
        assert!(!feature.row_is_available(0));
        assert_eq!(feature.status, vec![FeatureStatus::Unsupported]);
    }
}

#[test]
fn labeled_cut_constructor_requires_a_sorted_nonempty_proper_subset() {
    assert!(LabeledCut::new(3, vec![1, 0]).is_err());
    assert!(LabeledCut::new(3, vec![]).is_err());
    assert!(LabeledCut::new(3, vec![0, 1, 2]).is_err());
    assert!(LabeledCut::new(3, vec![0, 0]).is_err());
}

#[test]
fn labeled_cut_relabel_maps_the_selected_physical_wires() {
    let cut = LabeledCut::new(4, vec![0, 2]).unwrap();

    let relabeled = cut.relabel(&[1, 3, 2, 0]).unwrap();

    assert_eq!(relabeled.side_a(), &[1, 2]);
    assert_eq!(relabeled.side_b(), vec![0, 3]);
    assert!(cut.relabel(&[1, 1, 2, 3]).is_err());
}

#[test]
fn labeled_cut_registration_does_not_leave_a_partial_provider_set() {
    let mut registry = FeatureRegistry::new();
    let mut c03 = feature_spec(FeatureId::new(b'C', 3).unwrap());
    c03.producer_version = "preexisting_c03_v1".into();
    c03.exactness = "exact".into();
    registry.register(Box::new(ExistingC03(c03))).unwrap();

    assert!(
        register_labeled_cuts(&mut registry, vec![LabeledCut::new(3, vec![0]).unwrap()]).is_err()
    );

    let specs = registry.specs();
    assert_eq!(
        specs[FeatureId::new(b'C', 2).unwrap().index()].producer_version,
        "unconfigured"
    );
    assert_eq!(
        specs[FeatureId::new(b'C', 3).unwrap().index()].producer_version,
        "preexisting_c03_v1"
    );
}

#[test]
fn restricted_subspace_features_distinguish_row_column_and_candidate_rank() {
    // M = [[1,1,0],[0,1,1],[0,0,1]]. On coordinate {1}, rows 0 and 1
    // coincide, while columns 0 and 1 differ. Candidate e1 adds one rank to
    // row 0 restricted to coordinates {0,1}.
    let state =
        MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", vec![0b011, 0b110, 0b100])
            .unwrap();
    let intersection = RestrictedIntersection::new(3, vec![0], vec![1], vec![1]).unwrap();
    let increment = RankIncrementProbe::new(3, vec![0], vec![0, 1], vec![1]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_restricted_intersections(&mut registry, vec![intersection]).unwrap();
    register_rank_increment_probes(&mut registry, vec![increment]).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: [6, 7, 9]
            .into_iter()
            .map(|ordinal| FeatureId::new(b'C', ordinal).unwrap())
            .collect(),
        seed: 0,
        schema_id: "restricted-subspace-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };

    let packet = registry.build(&[state], &policy).unwrap();

    for (ordinal, expected) in [(6, 1.0), (7, 0.0), (9, 1.0)] {
        let feature = packet.feature(FeatureId::new(b'C', ordinal).unwrap());
        assert!(feature.row_is_available(0));
        assert_eq!(feature.values, vec![expected]);
        assert_eq!(feature.token_valid, Some(vec![1]));
    }
    for ordinal in [6, 7] {
        assert_eq!(
            packet
                .feature(FeatureId::new(b'C', ordinal).unwrap())
                .incidence,
            Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        );
    }
    assert_eq!(
        packet.feature(FeatureId::new(b'C', 9).unwrap()).incidence,
        Some(vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    );
}
