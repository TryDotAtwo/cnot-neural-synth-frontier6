use cnot103_native::{
    compute_h07_prediction_spread, FeatureId, FeatureStatus, H07Error, H07InputCoverage,
    H07InputCoverageEntry, H07ProbeKind, H07ProbePrediction,
};

fn sample(id: &str, kind: H07ProbeKind, prediction: f64) -> H07ProbePrediction {
    H07ProbePrediction {
        probe_id: id.into(),
        kind,
        prediction,
        input_coverage: None,
    }
}

fn probes() -> Vec<H07ProbePrediction> {
    vec![
        sample("base", H07ProbeKind::Canonical, 4.0),
        sample("inverse", H07ProbeKind::Inverse, 2.0),
        sample("permutation-01", H07ProbeKind::Relabel, 6.0),
    ]
}

#[test]
fn h07_is_a_post_forward_population_variance_with_bound_provenance() {
    let diagnostic = compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &probes())
        .expect("complete finite view/relabel probe set");

    assert_eq!(diagnostic.id, FeatureId::new(b'H', 7).unwrap());
    assert!((diagnostic.population_variance - 8.0 / 3.0).abs() < 1e-12);
    assert_eq!(diagnostic.range, 4.0);
    assert_eq!(diagnostic.mean, 4.0);
    assert_eq!(diagnostic.sample_count, 3);
    assert_eq!(diagnostic.view_sample_count, 2);
    assert_eq!(diagnostic.relabel_sample_count, 1);
    assert_eq!(diagnostic.model_contract_sha256, "1".repeat(64));
    assert_eq!(diagnostic.problem_scope_sha256, "2".repeat(64));
    assert_eq!(diagnostic.probes.len(), 3);
    assert_eq!(
        diagnostic.producer_version,
        "h07_distance_field_value_population_variance_v3"
    );
}

fn coverage(reason: &str) -> H07InputCoverage {
    H07InputCoverage {
        schema_id: "test-schema".into(),
        policy_id: "test-policy".into(),
        feature_id_manifest_id: "test-manifest".into(),
        entries: vec![
            H07InputCoverageEntry {
                id: FeatureId::new(b'A', 1).unwrap(),
                status: FeatureStatus::Computed,
                available: 1,
                reason: reason.into(),
            },
            H07InputCoverageEntry {
                id: FeatureId::new(b'A', 2).unwrap(),
                status: FeatureStatus::Inapplicable,
                available: 0,
                reason: "source-defined inapplicable result".into(),
            },
            H07InputCoverageEntry {
                id: FeatureId::new(b'A', 3).unwrap(),
                status: FeatureStatus::Unsupported,
                available: 0,
                reason: "producer has no result for this state".into(),
            },
        ],
    }
}

#[test]
fn h07_coverage_accepts_exact_canonical_102_input_statuses() {
    let h07 = FeatureId::new(b'H', 7).unwrap();
    let entries = (0..103)
        .filter_map(FeatureId::from_index)
        .filter(|id| *id != h07)
        .map(|id| H07InputCoverageEntry {
            id,
            status: if id == FeatureId::new(b'A', 2).unwrap() {
                FeatureStatus::Inapplicable
            } else {
                FeatureStatus::Computed
            },
            available: u8::from(id != FeatureId::new(b'A', 2).unwrap()),
            reason: "explicit test result".into(),
        })
        .collect::<Vec<_>>();
    let audit = H07InputCoverage {
        schema_id: "test-schema".into(),
        policy_id: "test-policy".into(),
        feature_id_manifest_id: "cnot_feature_ids_a01_i11_v1".into(),
        entries,
    };

    assert_eq!(audit.selected_model_input_count(), 102);
    assert_eq!(audit.computed_model_input_count(), 101);
    assert_eq!(audit.inapplicable_model_input_count(), 1);
    assert_eq!(audit.unavailable_model_input_count(), 0);
    assert!(audit.is_complete());
}

#[test]
fn h07_hash_binds_per_probe_input_coverage_and_reasons() {
    let mut original = probes();
    original[0].input_coverage = Some(coverage("computed from exact state"));
    let first = compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &original)
        .expect("coverage-bearing diagnostic");
    let audit = original[0].input_coverage.as_ref().unwrap();
    assert_eq!(audit.selected_model_input_count(), 3);
    assert_eq!(audit.computed_model_input_count(), 1);
    assert_eq!(audit.inapplicable_model_input_count(), 1);
    assert_eq!(audit.unavailable_model_input_count(), 1);
    assert!(!audit.is_complete());

    let mut changed = original.clone();
    changed[0].input_coverage.as_mut().unwrap().entries[0].reason =
        "different source result".into();
    let second = compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &changed)
        .expect("changed coverage-bearing diagnostic");
    assert_ne!(first.probe_set_sha256, second.probe_set_sha256);
}

#[test]
fn h07_rejects_missing_view_or_relabel_probes() {
    let view_only = vec![
        sample("base", H07ProbeKind::Canonical, 4.0),
        sample("inverse", H07ProbeKind::Inverse, 2.0),
    ];
    assert!(matches!(
        compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &view_only),
        Err(H07Error::MissingRelabelProbe)
    ));

    let relabel_only = vec![
        sample("base", H07ProbeKind::Canonical, 4.0),
        sample("permutation-01", H07ProbeKind::Relabel, 6.0),
    ];
    assert!(matches!(
        compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &relabel_only),
        Err(H07Error::MissingEquivalentViewProbe)
    ));
}

#[test]
fn h07_is_order_stable_and_rejects_invalid_probe_data() {
    let original = probes();
    let mut reordered = original.clone();
    reordered.reverse();
    let first = compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &original).unwrap();
    let second =
        compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &reordered).unwrap();
    assert_eq!(first.probe_set_sha256, second.probe_set_sha256);
    assert_eq!(first.population_variance, second.population_variance);

    let mut bad = probes();
    bad[1].prediction = f64::NAN;
    assert!(matches!(
        compute_h07_prediction_spread(&"1".repeat(64), &"2".repeat(64), &bad),
        Err(H07Error::NonFinitePrediction(_))
    ));

    assert!(matches!(
        compute_h07_prediction_spread("bad", &"2".repeat(64), &original),
        Err(H07Error::InvalidDigest(_))
    ));
}
