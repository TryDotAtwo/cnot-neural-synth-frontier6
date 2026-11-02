use cnot103_native::{
    register_symmetry_alignment_h06, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole,
    FeatureStatus, MatrixState,
};

fn policy(max_bytes: Option<usize>) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'H', 6).unwrap()],
        seed: 0,
        schema_id: "h06-source-pair-alignment-v1".into(),
        solver_assisted: false,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes,
    }
}

fn permute_state(state: &MatrixState, permutation: &[u32]) -> MatrixState {
    let n = state.dimension() as usize;
    let mut rows = vec![0u64; n * state.words_per_row()];
    for source_row in 0..n {
        for source_column in 0..n {
            let row = permutation[source_row] as usize;
            let column = permutation[source_column] as usize;
            if state.row_words(source_row as u32).unwrap()[source_column / 64]
                & (1u64 << (source_column % 64))
                != 0
            {
                rows[row * state.words_per_row() + column / 64] |= 1u64 << (column % 64);
            }
        }
    }
    MatrixState::from_packed_rows(state.dimension(), state.alphabet_id(), rows).unwrap()
}

#[test]
fn h06_matches_the_python_source_golden_for_all_four_views() {
    // Golden slices are from research_v3.features.symmetry_alignment on
    // MatrixState(3, (3,2,4), all_to_all_unit_cnot:3).
    let state = MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", vec![3, 2, 4]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_symmetry_alignment_h06(&mut registry).unwrap();
    let packet = registry.build(&[state], &policy(None)).unwrap();
    let id = FeatureId::new(b'H', 6).unwrap();
    let feature = packet.feature(id);
    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.spec.role, FeatureRole::Pair);
    assert_eq!(feature.spec.channels, 15);
    assert_eq!(feature.shape, vec![1, 3, 3, 15]);

    let pair = |row: usize, column: usize| {
        let start = (row * 3 + column) * 15;
        &feature.values[start..start + 15]
    };
    assert_eq!(
        pair(0, 0),
        &[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    );
    assert_eq!(
        pair(1, 0),
        &[0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]
    );
}

#[test]
fn h06_reports_byte_budget_exhaustion_without_emitting_values() {
    let state = MatrixState::identity(3, "all_to_all_unit_cnot:3").unwrap();
    let mut registry = FeatureRegistry::new();
    register_symmetry_alignment_h06(&mut registry).unwrap();
    let packet = registry.build(&[state], &policy(Some(1))).unwrap();
    let feature = packet.feature(FeatureId::new(b'H', 6).unwrap());
    assert_eq!(feature.status, vec![FeatureStatus::ResourceLimit]);
    assert_eq!(feature.available, vec![0]);
}

#[test]
fn h06_is_equivariant_under_physical_wire_relabeling_and_crosses_word_boundaries() {
    let state = MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", vec![3, 2, 4]).unwrap();
    let permutation = [2, 0, 1];
    let moved = permute_state(&state, &permutation);
    let mut registry = FeatureRegistry::new();
    register_symmetry_alignment_h06(&mut registry).unwrap();
    let original = registry.build(&[state.clone()], &policy(None)).unwrap();
    let relabeled = registry.build(&[moved], &policy(None)).unwrap();
    let id = FeatureId::new(b'H', 6).unwrap();
    let original = original.feature(id);
    let relabeled = relabeled.feature(id);
    for row in 0..3 {
        for column in 0..3 {
            let source = (row * 3 + column) * 15;
            let target = (permutation[row] as usize * 3 + permutation[column] as usize) * 15;
            assert_eq!(
                &relabeled.values[target..target + 15],
                &original.values[source..source + 15]
            );
        }
    }

    let mut wide = MatrixState::identity(65, "all_to_all_unit_cnot:65").unwrap();
    wide.apply_cnot(cnot103_native::CnotGate {
        control: 64,
        target: 0,
    })
    .unwrap();
    let wide_packet = registry.build(&[wide], &policy(None)).unwrap();
    let wide_feature = wide_packet.feature(id);
    assert_eq!(wide_feature.shape, vec![1, 65, 65, 15]);
    assert!(wide_feature.values.iter().all(|value| value.is_finite()));
}
