use cnot103_native::{
    compile_general_cnot_word, register_general_compiler_bounds, CnotAlphabet, FeatureId,
    FeaturePolicy, FeatureRegistry, FeatureRole, FeatureStatus, GeneralCompilerError,
    GeneralCompilerLimits, MatrixState,
};

fn bound_policy(ids: Vec<FeatureId>) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: ids,
        seed: 0,
        schema_id: "general-compiler-bounds-v1".into(),
        solver_assisted: true,
        symmetry_contract: "augmentation_only".into(),
        evaluations: Some(64),
        max_bytes: Some(1 << 20),
    }
}

fn registry_with_bounds() -> FeatureRegistry {
    let mut registry = FeatureRegistry::new();
    register_general_compiler_bounds(&mut registry).unwrap();
    registry
}

fn gap_one_state() -> MatrixState {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    // Rows [001, 011, 111]. Direct Gauss-Jordan compilation uses 3 gates;
    // the hand-counted changed-row/column/rank lower bound is 2.
    MatrixState::from_packed_rows(3, alphabet.id(), vec![1, 3, 7]).unwrap()
}

#[test]
fn f09_emits_verified_upper_bound_lower_bound_gap_and_tightness() {
    let f09 = FeatureId::new(b'F', 9).unwrap();
    let packet = registry_with_bounds()
        .build(&[gap_one_state()], &bound_policy(vec![f09]))
        .unwrap();
    let feature = packet.feature(f09);

    assert_eq!(feature.status, vec![FeatureStatus::Computed]);
    assert_eq!(feature.spec.role, FeatureRole::Global);
    assert_eq!(feature.spec.channels, 4);
    assert_eq!(feature.values, vec![3.0, 2.0, 1.0, 0.0]);
}

#[test]
fn f10_marks_optimality_only_when_verified_bounds_meet() {
    let f10 = FeatureId::new(b'F', 10).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let identity = MatrixState::identity(3, alphabet.id()).unwrap();
    let packet = registry_with_bounds()
        .build(&[gap_one_state(), identity], &bound_policy(vec![f10]))
        .unwrap();
    let feature = packet.feature(f10);

    assert_eq!(feature.status, vec![FeatureStatus::Computed; 2]);
    assert_eq!(feature.spec.role, FeatureRole::Global);
    assert_eq!(feature.spec.channels, 5);
    assert_eq!(
        feature.values,
        vec![
            1.0, 1.0, 0.0, 0.0, 1.0, // certified interval, compiler complete
            1.0, 1.0, 1.0, 0.0, 1.0, // tight bounds prove the identity optimum
        ]
    );
}

#[test]
fn f09_f10_require_explicit_solver_assisted_policy() {
    let f09 = FeatureId::new(b'F', 9).unwrap();
    let mut policy = bound_policy(vec![f09]);
    policy.solver_assisted = false;
    assert!(registry_with_bounds()
        .selected_specs(&policy)
        .unwrap_err()
        .to_string()
        .contains("requires solver-assisted policy"));
}

#[test]
fn feature_evaluation_budget_does_not_become_a_gate_budget() {
    let f09 = FeatureId::new(b'F', 9).unwrap();
    let mut policy = bound_policy(vec![f09]);
    policy.evaluations = Some(1);
    let packet = registry_with_bounds()
        .build(&[gap_one_state()], &policy)
        .unwrap();

    assert_eq!(packet.feature(f09).status, vec![FeatureStatus::Computed]);
    assert_eq!(packet.feature(f09).values, vec![3.0, 2.0, 1.0, 0.0]);
}

#[test]
fn general_compiler_replays_all_invertible_three_by_three_matrices() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let limits = GeneralCompilerLimits::default();
    let mut covered = 0usize;
    for packed in 0u16..(1 << 9) {
        let rows = (0..3)
            .map(|row| u64::from((packed >> (row * 3)) & 0b111))
            .collect::<Vec<_>>();
        let Ok(state) = MatrixState::from_packed_rows(3, alphabet.id(), rows) else {
            // MatrixState enforces its invertibility invariant at construction.
            continue;
        };
        let certificate = compile_general_cnot_word(&state, limits).unwrap();
        assert_eq!(certificate.verify(&alphabet).unwrap(), state);
        assert_eq!(certificate.physical_cost, certificate.gates.len() as u64);
        covered += 1;
    }
    assert_eq!(covered, 168);
}

#[test]
fn general_compiler_fails_closed_when_gate_limit_is_too_small() {
    let alphabet = CnotAlphabet::all_to_all_unit(2).unwrap();
    let swap = MatrixState::from_packed_rows(2, alphabet.id(), vec![2, 1]).unwrap();
    assert!(matches!(
        compile_general_cnot_word(
            &swap,
            GeneralCompilerLimits {
                max_dimension: 2,
                max_gates: 2,
            }
        ),
        Err(GeneralCompilerError::ResourceLimit(_))
    ));
}
