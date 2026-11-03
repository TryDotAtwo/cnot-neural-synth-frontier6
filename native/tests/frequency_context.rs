use cnot103_native::{
    register_frequency_contexts, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole,
    FeatureStatus, FrequencyContext, MatrixState,
};
use std::collections::HashSet;

fn id(text: &str) -> FeatureId {
    FeatureId::new(text.as_bytes()[0], text[1..].parse().unwrap()).unwrap()
}

fn policy(symmetry_contract: &str) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id("I07")],
        seed: 0,
        schema_id: "i07-stage52-source-fixture-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn state(n: u32, rows: Vec<u64>) -> MatrixState {
    MatrixState::from_packed_rows(n, &format!("all_to_all_unit_cnot:{n}"), rows).unwrap()
}

#[test]
fn i07_matches_source_fourier_rank_for_all_relabelled_2x2_frequencies() {
    // Same noncanonical physical partition and nonzero A/C/D blocks as the
    // research-v3 test_frequency_context.py source fixture.
    let order = [3u32, 0, 4, 1, 2];
    let canonical_rows = [1u64, 2, 7, 13, 23];
    let mut physical_rows = vec![0u64; 5];
    for (row_index, physical_row) in order.iter().copied().enumerate() {
        for (column_index, physical_column) in order.iter().copied().enumerate() {
            if canonical_rows[row_index] & (1 << column_index) != 0 {
                physical_rows[physical_row as usize] |= 1 << physical_column;
            }
        }
    }
    let state = state(5, physical_rows);

    for first_frequency_row in 0u64..4 {
        for second_frequency_row in 0u64..4 {
            let context = FrequencyContext::new(
                5,
                order[..2].to_vec(),
                order[2..3].to_vec(),
                order[3..].to_vec(),
                vec![vec![first_frequency_row], vec![second_frequency_row]],
            )
            .unwrap();
            let image: HashSet<usize> = (0usize..4)
                .map(|input| {
                    let first = ((first_frequency_row as usize & input).count_ones() & 1) as usize;
                    let second =
                        ((second_frequency_row as usize & input).count_ones() & 1) as usize;
                    first | (second << 1)
                })
                .collect();
            let kappa = image.len().ilog2() as usize;

            let mut registry = FeatureRegistry::new();
            register_frequency_contexts(&mut registry, vec![context], 144, 20_736).unwrap();
            let packet = registry
                .build(&[state.clone()], &policy("augmentation_only"))
                .unwrap();
            let feature = packet.feature(id("I07"));

            assert_eq!(feature.spec.role, FeatureRole::Global);
            assert_eq!(feature.spec.channels, 2);
            assert_eq!(feature.shape, vec![1, 2]);
            assert_eq!(feature.status, vec![FeatureStatus::Computed]);
            assert_eq!(feature.values, vec![kappa as f32, kappa as f32]);
        }
    }
}

#[test]
fn i07_matches_source_wide_rank_and_exact_representation_exponent() {
    let n = 144u32;
    let context = FrequencyContext::new(
        n,
        (0..48).collect(),
        (48..96).collect(),
        (96..144).collect(),
        (0..48).map(|bit| vec![1u64 << bit]).collect(),
    )
    .unwrap();
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, vec![context], 144, 20_736).unwrap();
    let packet = registry
        .build(
            &[MatrixState::identity(n, &format!("all_to_all_unit_cnot:{n}")).unwrap()],
            &policy("augmentation_only"),
        )
        .unwrap();
    let feature = packet.feature(id("I07"));
    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.values, vec![48.0, 2304.0]);

    // The central frequency row crosses the u64 boundary at source wire 64.
    let high_limb =
        FrequencyContext::new(67, (0..65).collect(), vec![65], vec![66], vec![vec![0, 1]]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, vec![high_limb], 144, 20_736).unwrap();
    let packet = registry
        .build(
            &[MatrixState::identity(67, "all_to_all_unit_cnot:67").unwrap()],
            &policy("augmentation_only"),
        )
        .unwrap();
    assert_eq!(packet.feature(id("I07")).values, vec![1.0, 1.0]);
}

#[test]
fn i07_distinguishes_interface_inapplicability_from_alphabet_and_budget_failures() {
    let context = FrequencyContext::new(3, vec![0], vec![1], vec![2], vec![vec![1]]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, vec![context.clone()], 2, 2).unwrap();
    let target = state(3, vec![3, 2, 4]);
    let limited = registry
        .build(&[target.clone()], &policy("augmentation_only"))
        .unwrap();
    assert_eq!(
        limited.feature(id("I07")).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, vec![context], 3, 2).unwrap();
    let outside_interface = registry
        .build(&[target], &policy("augmentation_only"))
        .unwrap();
    assert_eq!(
        outside_interface.feature(id("I07")).status,
        vec![FeatureStatus::Inapplicable]
    );

    let wrong_alphabet = MatrixState::identity(3, "custom_cnot").unwrap();
    let unsupported = registry
        .build(&[wrong_alphabet], &policy("augmentation_only"))
        .unwrap();
    assert_eq!(
        unsupported.feature(id("I07")).status,
        vec![FeatureStatus::Unsupported]
    );
    assert!(registry.selected_specs(&policy("exact_relabel")).is_err());
}

#[test]
fn i07_is_inapplicable_outside_the_explicit_interface() {
    let context = FrequencyContext::new(4, vec![0], vec![1, 2], vec![3], vec![vec![1]]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_frequency_contexts(&mut registry, vec![context], 4, 4).unwrap();

    // CNOT 3 -> 0 breaks the source-wire identity constraint of stage-52.
    let outside = state(4, vec![9, 2, 4, 8]);
    let packet = registry
        .build(&[outside], &policy("augmentation_only"))
        .unwrap();
    let feature = packet.feature(id("I07"));
    assert_eq!(feature.status, vec![FeatureStatus::Inapplicable]);
    assert_eq!(feature.available, vec![0]);
}

#[test]
fn i07_context_rejects_invalid_physical_partitions_and_frequency_padding() {
    assert!(FrequencyContext::new(3, vec![0], vec![1], vec![2], vec![vec![2]]).is_err());
    assert!(FrequencyContext::new(3, vec![0], vec![0], vec![2], vec![vec![1]]).is_err());
    assert!(FrequencyContext::new(3, vec![0], vec![1], vec![2], vec![]).is_err());
}
