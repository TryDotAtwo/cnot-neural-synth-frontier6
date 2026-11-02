use cnot103_native::{
    compute_i09_response, register_i09_quadratic_response, FeatureId, FeaturePolicy,
    FeatureRegistry, FeatureRole, FeatureStatus, I09Context, I09Limits, I09Monomial, MatrixState,
    I09_STAGE60_SOURCE_SHA256,
};

fn id(text: &str) -> FeatureId {
    FeatureId::new(text.as_bytes()[0], text[1..].parse().unwrap()).unwrap()
}

fn policy(symmetry_contract: &str) -> FeaturePolicy {
    FeaturePolicy {
        enabled_ids: vec![id("I09")],
        seed: 0,
        schema_id: "i09-stage60-source-fixture-v1".into(),
        solver_assisted: false,
        symmetry_contract: symmetry_contract.into(),
        evaluations: None,
        max_bytes: None,
    }
}

fn state(rows: Vec<u64>) -> MatrixState {
    MatrixState::from_packed_rows(2, "all_to_all_unit_cnot:2", rows).unwrap()
}

fn context(prefix_bits: Vec<u8>) -> I09Context {
    I09Context::new(2, 1, 3, prefix_bits, vec![(0, 1), (1, 0)]).unwrap()
}

fn fixture_registry(context: I09Context, limits: I09Limits) -> FeatureRegistry {
    let mut registry = FeatureRegistry::new();
    register_i09_quadratic_response(&mut registry, vec![context], limits).unwrap();
    registry
}

fn evaluate_equation(equation: &[I09Monomial], assignment: usize) -> u8 {
    equation.iter().fold(0, |value, monomial| {
        let term = match *monomial {
            I09Monomial::Constant => 1,
            I09Monomial::Variable(index) => ((assignment >> index) & 1) as u8,
            I09Monomial::Product(left, right) => {
                (((assignment >> left) & 1) & ((assignment >> right) & 1)) as u8
            }
        };
        value ^ term
    })
}

fn direct_word_assignment(word: Option<(usize, usize)>) -> (Vec<u64>, usize) {
    let mut rows = vec![1u64, 2];
    let mut assignment = 0usize;
    let mut readout = 0u64;
    if let Some((control, target)) = word {
        assignment |= 1; // active[0]
        assignment |= 1 << (1 + control);
        assignment |= 1 << (3 + target);
        readout = rows[control];
        rows[target] ^= readout;
    }
    for bit in 0..2 {
        assignment |= (((readout >> bit) & 1) as usize) << (5 + bit);
    }
    for row in 0..2 {
        for column in 0..2 {
            assignment |= (((rows[row] >> column) & 1) as usize) << (7 + 2 * row + column);
        }
    }
    (rows, assignment)
}

#[test]
fn i09_compiler_equations_match_direct_one_gate_word_replay() {
    let words = [None, Some((0, 1)), Some((1, 0))];
    for target in [vec![1, 2], vec![1, 3], vec![3, 2], vec![2, 1]] {
        let response = compute_i09_response(
            &state(target.clone()),
            &context(vec![0, 0, 0]),
            I09Limits::default(),
        )
        .unwrap();
        assert_eq!(response.variable_names.len(), 11);
        assert_eq!(response.equations.len(), 16);

        let expected = words
            .iter()
            .filter_map(|word| {
                let (rows, assignment) = direct_word_assignment(*word);
                (rows == target).then_some(assignment)
            })
            .collect::<std::collections::HashSet<_>>();
        let observed = (0usize..(1 << response.variable_names.len()))
            .filter(|assignment| {
                response
                    .equations
                    .iter()
                    .all(|equation| evaluate_equation(equation, *assignment) == 0)
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(observed, expected);
    }
}

#[test]
fn i09_zero_horizon_keeps_the_initial_matrix_boundary_equations() {
    let empty_path = I09Context::new(2, 0, 0, vec![], vec![(0, 1), (1, 0)]).unwrap();
    let limits = I09Limits {
        max_variables: 11,
        max_equations: 19,
        max_bytes: 8_192,
    };
    let identity = compute_i09_response(&state(vec![1, 2]), &empty_path, limits).unwrap();
    assert!(identity.variable_names.is_empty());
    assert_eq!(identity.equations.len(), 4);
    assert!(identity.equations.iter().all(Vec::is_empty));
    assert_eq!(identity.alpha, vec![0; 4]);
    assert_eq!(identity.rank_b, 0);
    assert_eq!(identity.mixed_rank, 0);

    let unreachable = compute_i09_response(&state(vec![1, 3]), &empty_path, limits).unwrap();
    assert_eq!(unreachable.equations.len(), 4);
    assert_eq!(unreachable.equations[2], vec![I09Monomial::Constant]);
    assert_eq!(unreachable.alpha, vec![0, 0, 1, 0]);
    assert!(unreachable
        .equations
        .iter()
        .enumerate()
        .all(|(index, equation)| index == 2 || equation.is_empty()));
}

#[test]
fn i09_matches_pinned_stage60_golden_for_every_gl2_target() {
    assert_eq!(
        I09_STAGE60_SOURCE_SHA256,
        "fcdaec83bff7c111261dfc46c2040ddbbc349dbbc11a712a31af945dca3c6339"
    );
    let registry = fixture_registry(
        context(vec![0, 1, 1]),
        I09Limits {
            max_variables: 11,
            max_equations: 19,
            max_bytes: 8_192,
        },
    );

    // Literal Stage-60 source outputs for all four GL(2,2) targets, cut=3,
    // little-endian prefix 6, and the full ordered two-wire CNOT alphabet.
    let targets = [
        (vec![1, 2], 39_364u32),
        (vec![1, 3], 55_748),
        (vec![3, 2], 47_556),
        (vec![2, 1], 27_076),
    ];
    for (rows, expected_alpha) in targets {
        let packet = registry
            .build(&[state(rows)], &policy("augmentation_only"))
            .unwrap();
        let feature = packet.feature(id("I09"));
        assert_eq!(feature.spec.role, FeatureRole::Group);
        assert_eq!(feature.spec.channels, 15);
        assert_eq!(feature.shape, vec![1, 16, 15]);
        assert_eq!(feature.status, vec![FeatureStatus::Computed]);
        assert_eq!(feature.available, vec![1]);
        assert_eq!(feature.provenance.len(), 1);
        assert!(feature.provenance[0][0].contains(I09_STAGE60_SOURCE_SHA256));
        assert_eq!(feature.token_valid, Some(vec![1; 16]));

        let row = |equation: usize| &feature.values[equation * 15..(equation + 1) * 15];
        for equation in 0..16 {
            assert_eq!(row(equation)[0], equation as f32);
            assert_eq!(row(equation)[1], ((expected_alpha >> equation) & 1) as f32);
            assert_eq!(row(equation)[2], 2.0); // rank(B)
            assert_eq!(row(equation)[3], 2.0); // mixed rank
            assert_eq!(row(equation)[4], if equation == 4 { 1.0 } else { 0.0 });
            assert_eq!(row(equation)[5], if equation == 5 { 1.0 } else { 0.0 });
            assert!(row(equation)[6..].iter().all(|value| *value == 0.0));
        }

        let incidence = feature.incidence.as_ref().unwrap();
        assert_eq!(incidence.len(), 16 * 2);
        assert_eq!(&incidence[12 * 2..13 * 2], &[1.0, 0.0]);
        assert_eq!(&incidence[14 * 2..15 * 2], &[1.0, 1.0]);
        assert_eq!(&incidence[15 * 2..16 * 2], &[0.0, 1.0]);
    }
}

#[test]
fn i09_fails_closed_for_context_width_alphabet_and_resource_limits() {
    let limited = fixture_registry(
        context(vec![0, 0, 0]),
        I09Limits {
            max_variables: 10,
            max_equations: 19,
            max_bytes: 8_192,
        },
    );
    let packet = limited
        .build(&[state(vec![1, 3])], &policy("augmentation_only"))
        .unwrap();
    assert_eq!(
        packet.feature(id("I09")).status,
        vec![FeatureStatus::ResourceLimit]
    );

    let registry = fixture_registry(
        context(vec![0, 0, 0]),
        I09Limits {
            max_variables: 11,
            max_equations: 19,
            max_bytes: 8_192,
        },
    );
    let wrong_alphabet = MatrixState::identity(2, "custom_cnot").unwrap();
    let packet = registry
        .build(&[wrong_alphabet], &policy("augmentation_only"))
        .unwrap();
    assert_eq!(
        packet.feature(id("I09")).status,
        vec![FeatureStatus::Unsupported]
    );

    assert!(registry.selected_specs(&policy("exact_relabel")).is_err());
}

#[test]
fn i09_context_and_registration_reject_ambiguous_contracts() {
    assert!(I09Context::new(0, 0, 0, vec![], vec![]).is_err());
    assert!(I09Context::new(2, 1, 3, vec![0, 2, 0], vec![(0, 1)]).is_err());
    assert!(I09Context::new(2, 1, 3, vec![0, 0], vec![(0, 1)]).is_err());
    assert!(I09Context::new(2, 1, 3, vec![0, 0, 0], vec![(1, 0), (0, 1)]).is_err());
    assert!(I09Context::new(2, 1, 3, vec![0, 0, 0], vec![(0, 1), (0, 1)]).is_err());
    assert!(I09Context::new(2, 1, 3, vec![0, 0, 0], vec![(0, 2)]).is_err());

    let duplicate_widths = vec![context(vec![0, 0, 0]), context(vec![1, 0, 0])];
    let mut registry = FeatureRegistry::new();
    assert!(
        register_i09_quadratic_response(&mut registry, duplicate_widths, I09Limits::default())
            .is_err()
    );
}
