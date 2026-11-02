use cnot103_native::{
    compute_h07_prediction_spread, FeatureId, H07Error, H07ProbeKind, H07ProbePrediction,
};

fn sample(id: &str, kind: H07ProbeKind, prediction: f64) -> H07ProbePrediction {
    H07ProbePrediction {
        probe_id: id.into(),
        kind,
        prediction,
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
        "h07_distance_field_value_population_variance_v2"
    );
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
