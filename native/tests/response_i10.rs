use cnot103_native::{
    CnotAlphabet, FeatureId, FeaturePolicy, FeatureRegistry, FeatureStatus, MatrixState,
};

fn policy(id: FeatureId) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "i10-exact-response-test-v1".into(),
        solver_assisted: true,
        symmetry_contract: "exact_relabel".into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn registry(
    max_n: u32,
    max_degree: usize,
    max_layer_states: usize,
    max_transitions: usize,
) -> FeatureRegistry {
    let mut registry = FeatureRegistry::new();
    cnot103_native::register_tiny_i10(
        &mut registry,
        cnot103_native::I10Config {
            max_n,
            max_degree,
            max_layer_states,
            max_transitions,
        },
    )
    .unwrap();
    registry
}

#[test]
fn i10_identity_has_the_source_degree_zero_leader() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let state = MatrixState::identity(2, alphabet.id()).unwrap();
    let packet = registry(2, 2, 32, 128)
        .build(&[state], &policy(id))
        .unwrap();
    let feature = packet.feature(id);

    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.available, vec![1]);
    assert_eq!(feature.values[0], 0.0);
    assert!((feature.values[1] - 2.0f32.ln()).abs() < 1e-6);
    assert_eq!(&feature.values[2..], &[1.0, 0.0]);
}

#[test]
fn i10_one_cnot_endpoint_has_exact_positive_coefficient_one() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    state
        .apply_cnot(cnot103_native::CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    let packet = registry(3, 3, 256, 4096)
        .build(&[state], &policy(id))
        .unwrap();
    let feature = packet.feature(id);

    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.available, vec![1]);
    assert_eq!(feature.values[0], 1.0);
    assert!((feature.values[1] - 2.0f32.ln()).abs() < 1e-6);
    assert_eq!(&feature.values[2..], &[1.0, 0.0]);
}

#[test]
fn i10_counts_distinct_minimum_words_with_multiplicity() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(cnot103_native::CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    state
        .apply_cnot(cnot103_native::CnotGate {
            control: 2,
            target: 3,
        })
        .unwrap();
    let packet = registry(4, 3, 5_000, 10_000)
        .build(&[state], &policy(id))
        .unwrap();
    let feature = packet.feature(id);

    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.values[0], 2.0);
    assert!((feature.values[1] - 3.0f32.ln()).abs() < 1e-6);
}

#[test]
fn i10_matches_source_two_gate_fixture_and_wire_relabeling() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    for (control, target) in [(0, 1), (1, 2)] {
        state
            .apply_cnot(cnot103_native::CnotGate { control, target })
            .unwrap();
    }
    let permutation = [2usize, 0, 1];
    let mut relabeled_rows = vec![0u64; 3];
    for old_row in 0..3 {
        for old_column in 0..3 {
            let bit = (state.packed_rows()[old_row] >> old_column) & 1;
            relabeled_rows[permutation[old_row]] |= bit << permutation[old_column];
        }
    }
    let relabeled = MatrixState::from_packed_rows(3, alphabet.id(), relabeled_rows).unwrap();
    let packet = registry(3, 5, 500, 10_000)
        .build(&[state, relabeled], &policy(id))
        .unwrap();
    let feature = packet.feature(id);

    assert_eq!(feature.status, vec![FeatureStatus::Computed; 2]);
    assert_eq!(&feature.values[0..4], &[2.0, 2.0f32.ln(), 1.0, 0.0]);
    assert_eq!(&feature.values[0..4], &feature.values[4..8]);
}

#[test]
fn i10_oracle_exposes_exact_series_scope_and_transition_audit() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    for (control, target) in [(0, 1), (1, 2)] {
        state
            .apply_cnot(cnot103_native::CnotGate { control, target })
            .unwrap();
    }
    let result = cnot103_native::evaluate_i10(
        &state,
        cnot103_native::I10Config {
            max_n: 3,
            max_degree: 5,
            max_layer_states: 500,
            max_transitions: 10_000,
        },
    );

    assert_eq!(result.status, cnot103_native::I10OracleStatus::Exact);
    assert_eq!(
        result.positive_terms,
        vec![(2, 1), (3, 3), (4, 23), (5, 86)]
    );
    assert_eq!(result.scope, "full_cnot_alphabet:n3");
    assert_eq!(result.evaluations, 1476);
    assert_eq!(result.reason, "complete_through_requested_degree");
}

#[test]
fn i10_requires_explicit_solver_assisted_policy() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let state = MatrixState::identity(2, alphabet.id()).unwrap();
    let registry = registry(2, 2, 32, 128);
    let mut raw_policy = policy(id);
    raw_policy.solver_assisted = false;

    assert!(registry.build(&[state], &raw_policy).is_err());
}

#[test]
fn i10_rejects_width_below_the_reference_domain() {
    let alphabet = CnotAlphabet::all_to_all_unit(1).unwrap();
    let state = MatrixState::identity(1, alphabet.id()).unwrap();
    let result = cnot103_native::evaluate_i10(
        &state,
        cnot103_native::I10Config {
            max_n: 3,
            max_degree: 1,
            max_layer_states: 10,
            max_transitions: 10,
        },
    );

    assert_eq!(result.status, cnot103_native::I10OracleStatus::Unsupported);
    assert_eq!(result.reason, "response_requires_width_at_least_two");
    assert!(result.positive_terms.is_empty());
}

#[test]
fn i10_does_not_turn_a_horizon_miss_into_an_exact_zero() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let mut state = MatrixState::identity(2, alphabet.id()).unwrap();
    state
        .apply_cnot(cnot103_native::CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let packet = registry(2, 0, 32, 128)
        .build(&[state], &policy(id))
        .unwrap();
    let feature = packet.feature(id);

    assert_eq!(feature.status, vec![FeatureStatus::Unsupported]);
    assert_eq!(feature.available, vec![0]);
    assert!(feature.reason[0].contains("no_positive_term_within_horizon"));
}

#[test]
fn i10_reports_a_resource_limit_without_emitting_partial_values() {
    let id = FeatureId::new(b'I', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let mut state = MatrixState::identity(2, alphabet.id()).unwrap();
    state
        .apply_cnot(cnot103_native::CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let oracle = cnot103_native::evaluate_i10(
        &state,
        cnot103_native::I10Config {
            max_n: 2,
            max_degree: 2,
            max_layer_states: 1,
            max_transitions: 128,
        },
    );
    let packet = registry(2, 2, 1, 128).build(&[state], &policy(id)).unwrap();
    let feature = packet.feature(id);

    assert_eq!(
        oracle.status,
        cnot103_native::I10OracleStatus::ResourceLimit
    );
    assert_eq!(oracle.evaluations, 2);
    assert!(oracle.positive_terms.is_empty());
    assert_eq!(oracle.scope, "full_cnot_alphabet:n2");
    assert_eq!(feature.status, vec![FeatureStatus::ResourceLimit]);
    assert_eq!(feature.available, vec![0]);
    assert!(feature.reason[0].contains("complete_next_layer_exceeds_declared_budget"));
}
