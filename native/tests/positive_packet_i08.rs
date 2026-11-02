use cnot103_native::{
    register_positive_packet_contexts, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole,
    FeatureStatus, LocalBooleanFactor, MatrixState, OptionalGatePacket,
};
use num_bigint::BigUint;

fn id(text: &str) -> FeatureId {
    FeatureId::new(text.as_bytes()[0], text[1..].parse().unwrap()).unwrap()
}

fn policy(symmetry_contract: &str, max_bytes: Option<usize>) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id("I08")],
        seed: 0,
        schema_id: "i08-stage59-fixture-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes,
    }
}

fn matrix(n: u32, alphabet: &str, rows: &[u64]) -> MatrixState {
    MatrixState::from_packed_rows(n, alphabet, rows.to_vec()).unwrap()
}

#[test]
fn i08_matches_stage59_cut_widths_live_selectors_and_physical_incidence() {
    let context = OptionalGatePacket::new(
        3,
        vec![(0, 2), (1, 2), (0, 2), (1, 2)],
        vec![
            LocalBooleanFactor::new(vec![0, 3], vec![BigUint::from(1u8), BigUint::from(2u8)])
                .unwrap(),
        ],
    )
    .unwrap();
    let mut registry = FeatureRegistry::new();
    register_positive_packet_contexts(&mut registry, vec![context], 144, 256, 4096, 65_536)
        .unwrap();
    let target = matrix(3, "all_to_all_unit_cnot:3", &[1, 2, 7]);
    let feature = registry
        .build(&[target], &policy("augmentation_only", None))
        .unwrap();
    let feature = feature.feature(id("I08"));

    assert_eq!(feature.spec.role, FeatureRole::Group);
    assert_eq!(feature.spec.channels, 4);
    assert_eq!(feature.shape, vec![1, 5, 4]);
    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.token_valid.as_deref(), Some(&[1, 1, 1, 1, 1][..]));
    assert_eq!(
        feature.values,
        vec![
            0.0, 0.0, 0.0, 0.0, // cut, kappa, live, kappa + live
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 0.0,
        ]
    );
    let incidence = feature.incidence.as_deref().unwrap();
    for cut in 1..=3 {
        assert_eq!(incidence[(cut * 3) * 2], 1.0); // gate 0 control, wire 0
        assert_eq!(incidence[(cut * 3 + 2) * 2 + 1], 1.0); // gate 0 target, wire 2
    }
    assert!(incidence[..6].iter().all(|value| *value == 0.0));
    assert!(incidence[24..].iter().all(|value| *value == 0.0));
}

#[test]
fn i08_rejects_non_affine_maps_and_reports_resource_limits_separately() {
    let non_affine = OptionalGatePacket::new(3, vec![(0, 1), (1, 2)], vec![]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_positive_packet_contexts(&mut registry, vec![non_affine.clone()], 3, 2, 4096, 65_536)
        .unwrap();
    let target = matrix(3, "all_to_all_unit_cnot:3", &[1, 2, 4]);
    let feature = registry
        .build(&[target.clone()], &policy("augmentation_only", None))
        .unwrap();
    assert_eq!(
        feature.feature(id("I08")).status,
        vec![FeatureStatus::Unsupported]
    );

    let mut registry = FeatureRegistry::new();
    register_positive_packet_contexts(&mut registry, vec![non_affine], 3, 2, 1, 65_536).unwrap();
    let feature = registry
        .build(&[target], &policy("augmentation_only", None))
        .unwrap();
    assert_eq!(
        feature.feature(id("I08")).status,
        vec![FeatureStatus::ResourceLimit]
    );
}

#[test]
fn i08_preserves_selector_labels_beyond_one_machine_word() {
    let gates = vec![(0, 1); 71];
    let factor =
        LocalBooleanFactor::new(vec![69, 70], vec![BigUint::from(1u8), BigUint::from(2u8)])
            .unwrap();
    let context = OptionalGatePacket::new(2, gates, vec![factor]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_positive_packet_contexts(&mut registry, vec![context], 2, 80, 4096, 65_536).unwrap();
    let target = matrix(2, "all_to_all_unit_cnot:2", &[1, 2]);
    let feature = registry
        .build(&[target], &policy("augmentation_only", None))
        .unwrap();
    let feature = feature.feature(id("I08"));

    assert_eq!(feature.shape, vec![1, 72, 4]);
    assert_eq!(&feature.values[..4], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&feature.values[69 * 4..70 * 4], &[69.0, 1.0, 0.0, 1.0]);
    assert_eq!(&feature.values[70 * 4..71 * 4], &[70.0, 1.0, 1.0, 2.0]);
    assert_eq!(&feature.values[71 * 4..], &[71.0, 0.0, 0.0, 0.0]);
    let incidence = feature.incidence.as_deref().unwrap();
    assert_eq!(incidence[(70 * 2) * 2], 1.0);
    assert_eq!(incidence[(70 * 2 + 1) * 2 + 1], 1.0);
}

#[test]
fn i08_fails_closed_for_context_policy_and_working_byte_budgets() {
    let context = OptionalGatePacket::new(2, vec![(0, 1)], vec![]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_positive_packet_contexts(&mut registry, vec![context], 2, 8, 4096, 65_536).unwrap();
    let target = matrix(2, "all_to_all_unit_cnot:2", &[1, 2]);

    assert!(registry
        .build(&[target.clone()], &policy("exact_relabel", None))
        .is_err());
    let feature = registry
        .build(&[target.clone()], &policy("augmentation_only", Some(1)))
        .unwrap();
    assert_eq!(
        feature.feature(id("I08")).status,
        vec![FeatureStatus::ResourceLimit]
    );
    let wrong_alphabet = matrix(2, "custom_cnot", &[1, 2]);
    let feature = registry
        .build(&[wrong_alphabet], &policy("augmentation_only", None))
        .unwrap();
    assert_eq!(
        feature.feature(id("I08")).status,
        vec![FeatureStatus::Unsupported]
    );
}

#[test]
fn i08_rejects_malformed_local_factor_contracts_and_duplicate_widths() {
    assert!(LocalBooleanFactor::new(vec![1, 1], vec![BigUint::from(0u8)]).is_err());
    assert!(LocalBooleanFactor::new(vec![0], vec![BigUint::from(2u8)]).is_err());
    assert!(
        LocalBooleanFactor::new(vec![0], vec![BigUint::from(1u8), BigUint::from(0u8)]).is_err()
    );
    assert!(OptionalGatePacket::new(2, vec![(0, 0)], vec![]).is_err());
    assert!(OptionalGatePacket::new(0, vec![], vec![]).is_err());

    let context = OptionalGatePacket::new(2, vec![(0, 1)], vec![]).unwrap();
    let duplicate = OptionalGatePacket::new(2, vec![(1, 0)], vec![]).unwrap();
    let mut registry = FeatureRegistry::new();
    assert!(register_positive_packet_contexts(
        &mut registry,
        vec![context, duplicate],
        2,
        8,
        4096,
        65_536,
    )
    .is_err());
}
