use cnot103_native::{
    register_shared_relation_contexts, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole,
    FeatureStatus, MatrixState, SharedRelationContext,
};

fn id(text: &str) -> FeatureId {
    FeatureId::new(text.as_bytes()[0], text[1..].parse().unwrap()).unwrap()
}

fn policy() -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id("I06")],
        seed: 0,
        schema_id: "i06-stage56-fixture-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn gf2_rank(rows: &[u64], width: usize) -> usize {
    let mut rows = rows.to_vec();
    let mut pivot_row = 0;
    for column in 0..width {
        let Some(found) = (pivot_row..rows.len()).find(|row| rows[*row] & (1u64 << column) != 0)
        else {
            continue;
        };
        rows.swap(pivot_row, found);
        let pivot = rows[pivot_row];
        for row in 0..rows.len() {
            if row != pivot_row && rows[row] & (1u64 << column) != 0 {
                rows[row] ^= pivot;
            }
        }
        pivot_row += 1;
    }
    pivot_row
}

fn span_masks(basis: &[u64]) -> std::collections::HashSet<u64> {
    let mut span = std::collections::HashSet::from([0]);
    for vector in basis {
        let previous: Vec<_> = span.iter().copied().collect();
        span.extend(previous.into_iter().map(|value| value ^ vector));
    }
    span
}

#[test]
fn i06_emits_the_full_labelled_kernel_ports_and_physical_incidence() {
    let context =
        SharedRelationContext::new(3, vec![2, 0, 1], vec![1, 0], vec![vec![1], vec![2]]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(&mut registry, vec![context], 3, 2, 2).unwrap();

    let state = MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", vec![3, 2, 7]).unwrap();
    let packet = registry.build(&[state], &policy()).unwrap();
    let feature = packet.feature(id("I06"));

    assert_eq!(feature.spec.role, FeatureRole::Group);
    assert_eq!(feature.spec.channels, 5);
    assert_eq!(feature.shape, vec![1, 3, 5]);
    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(
        feature.values,
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,]
    );
    assert_eq!(feature.token_valid.as_deref(), Some(&[1, 1, 1][..]));
    assert_eq!(
        feature.incidence.as_deref(),
        Some(
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 0.0,
            ][..]
        )
    );
}

#[test]
fn i06_retains_labelled_ports_even_when_relation_kernels_match() {
    let context = SharedRelationContext::new(2, vec![0, 1], vec![0, 1], vec![vec![1]]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(&mut registry, vec![context], 2, 2, 1).unwrap();
    let a = MatrixState::from_packed_rows(2, "all_to_all_unit_cnot:2", vec![1, 2]).unwrap();
    let b = MatrixState::from_packed_rows(2, "all_to_all_unit_cnot:2", vec![2, 1]).unwrap();

    let packet = registry.build(&[a, b], &policy()).unwrap();
    let feature = packet.feature(id("I06"));
    assert_eq!(feature.shape, vec![2, 1, 4]);
    assert_eq!(
        feature.values,
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,]
    );
}

#[test]
fn i06_rejects_dependent_covectors_and_respects_vector_budget() {
    assert!(
        SharedRelationContext::new(2, vec![0, 1], vec![0, 1], vec![vec![1], vec![1]],).is_err()
    );

    let context = SharedRelationContext::new(2, vec![0, 1], vec![0, 1], vec![vec![1]]).unwrap();
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(&mut registry, vec![context], 1, 2, 1).unwrap();
    let state = MatrixState::from_packed_rows(2, "all_to_all_unit_cnot:2", vec![1, 2]).unwrap();
    let packet = registry.build(&[state], &policy()).unwrap();

    assert_eq!(
        packet.feature(id("I06")).status,
        vec![FeatureStatus::ResourceLimit]
    );
}

#[test]
fn i06_matches_independent_relation_enumeration_for_every_invertible_3x3_matrix() {
    let contexts =
        vec![
            SharedRelationContext::new(3, vec![2, 0, 1], vec![1, 0], vec![vec![1], vec![2]])
                .unwrap(),
        ];
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(&mut registry, contexts, 3, 2, 2).unwrap();
    let mut states = Vec::new();
    let mut reference_vectors = Vec::new();
    for a in 0u64..8 {
        for b in 0u64..8 {
            for c in 0u64..8 {
                let rows = [a, b, c];
                if gf2_rank(&rows, 3) != 3 {
                    continue;
                }
                let state =
                    MatrixState::from_packed_rows(3, "all_to_all_unit_cnot:3", rows.to_vec())
                        .unwrap();
                let selected = [2usize, 0, 1]
                    .map(|row| u64::from((rows[row] >> 1) & 1) | (u64::from(rows[row] & 1) << 1));
                states.push(state);
                reference_vectors.push(selected);
            }
        }
    }
    assert_eq!(states.len(), 168);
    let packet = registry.build(&states, &policy()).unwrap();
    let feature = packet.feature(id("I06"));
    let groups = feature.shape[1];
    let channels = feature.shape[2];
    let token_valid = feature.token_valid.as_deref().unwrap();

    for (batch, vectors) in reference_vectors.iter().enumerate() {
        let mut kernel = Vec::new();
        let mut ports = [0u64; 2];
        for group in 0..groups {
            if token_valid[batch * groups + group] == 0 {
                continue;
            }
            let base = (batch * groups + group) * channels;
            let kind = feature.values[base] as usize;
            let label = feature.values[base + 1] as usize;
            let mask = (0..3).fold(0u64, |mask, index| {
                if feature.values[base + 2 + index] == 1.0 {
                    mask | (1 << index)
                } else {
                    mask
                }
            });
            if kind == 0 {
                kernel.push(mask);
            } else {
                ports[label] = mask;
            }
        }

        let actual_relations = span_masks(&kernel);
        let expected_relations: std::collections::HashSet<u64> = (0..8)
            .filter(|mask| {
                vectors
                    .iter()
                    .enumerate()
                    .fold(0u64, |sum, (index, vector)| {
                        if mask & (1 << index) != 0 {
                            sum ^ vector
                        } else {
                            sum
                        }
                    })
                    == 0
            })
            .collect();
        assert_eq!(actual_relations, expected_relations);
        assert_eq!(
            ports,
            [
                vectors
                    .iter()
                    .enumerate()
                    .fold(0u64, |mask, (index, vector)| mask | ((vector & 1) << index)),
                vectors
                    .iter()
                    .enumerate()
                    .fold(0u64, |mask, (index, vector)| {
                        mask | (((vector >> 1) & 1) << index)
                    }),
            ]
        );
    }
}

#[test]
fn i06_bit_packing_crosses_u64_boundaries_without_losing_labels() {
    let n = 70u32;
    let mut packed_rows = vec![0u64; n as usize * 2];
    for row in 0..n as usize {
        packed_rows[row * 2 + row / 64] |= 1u64 << (row % 64);
    }
    let state = MatrixState::from_packed_rows(n, "all_to_all_unit_cnot:70", packed_rows).unwrap();
    let context = SharedRelationContext::new(
        n,
        (0..65).collect(),
        (0..65).collect(),
        vec![vec![u64::MAX, 1]],
    )
    .unwrap();
    let mut registry = FeatureRegistry::new();
    register_shared_relation_contexts(&mut registry, vec![context], 65, 65, 1).unwrap();
    let packet = registry.build(&[state], &policy()).unwrap();
    let feature = packet.feature(id("I06"));

    assert_eq!(feature.shape, vec![1, 1, 67]);
    assert_eq!(&feature.values[..2], &[1.0, 0.0]);
    assert!(feature.values[2..].iter().all(|value| *value == 1.0));
    assert_eq!(feature.token_valid.as_deref(), Some(&[1][..]));
    let incidence = feature.incidence.as_deref().unwrap();
    for node in 0..70 {
        let expected = if node < 65 { [1.0, 1.0] } else { [0.0, 0.0] };
        assert_eq!(&incidence[node * 2..node * 2 + 2], &expected);
    }
}
