use cnot103_native::{
    register_template_response, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry,
    FeatureRole, FeatureStatus, MatrixState, RationalProbability, TemplateChoice,
    TemplateChoiceKind, TemplateResponseBudgets, TemplateResponseConfig,
};

fn e02_id() -> FeatureId {
    FeatureId::new(b'E', 2).unwrap()
}

#[test]
fn e_family_matches_exact_source_template_fixture() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();

    let choices = vec![
        TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        TemplateChoice::new(TemplateChoiceKind::Swap, 0, 1),
    ];
    let points = vec![
        vec![
            RationalProbability::new(1, 4).unwrap(),
            RationalProbability::new(3, 4).unwrap(),
        ],
        vec![
            RationalProbability::new(1, 3).unwrap(),
            RationalProbability::new(1, 4).unwrap(),
        ],
    ];
    let config = TemplateResponseConfig::new(choices, points).unwrap();
    let mut registry = FeatureRegistry::new();
    register_template_response(&mut registry, config).unwrap();

    let policy = FeaturePolicy {
        enabled_ids: [2, 3, 6, 7, 8, 9]
            .into_iter()
            .map(|ordinal| FeatureId::new(b'E', ordinal).unwrap())
            .collect(),
        seed: 0,
        schema_id: "template-response-family-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&[state], &policy).unwrap();
    let feature = packet.feature(e02_id());
    assert_eq!(feature.spec.role, FeatureRole::Global);
    assert_eq!(feature.spec.channels, 1);
    assert!(feature.row_is_available(0));
    // The four ordered assignments cost 1, 3, 7, and 9 CNOTs respectively.
    assert_eq!(feature.values, vec![5.0]);

    let e03 = packet.feature(FeatureId::new(b'E', 3).unwrap());
    assert!(e03.row_is_available(0));
    assert_eq!(e03.spec.channels, 2);
    assert_eq!(e03.values, vec![6.0, 19.0f32 / 6.0]);

    let e06 = packet.feature(FeatureId::new(b'E', 6).unwrap());
    assert!(e06.row_is_available(0));
    assert_eq!(e06.values, vec![0.0]);
    assert_eq!(e06.shape, vec![1, 1, 1]);
    assert_eq!(e06.token_valid.as_deref(), Some(&[1][..]));
    assert_eq!(e06.incidence.as_ref().map(Vec::len), Some(16));

    let e07 = packet.feature(FeatureId::new(b'E', 7).unwrap());
    assert!(e07.row_is_available(0));
    assert_eq!(e07.values, vec![1.0]);

    let e08 = packet.feature(FeatureId::new(b'E', 8).unwrap());
    assert!(e08.row_is_available(0));
    assert_eq!(e08.values, vec![1.0, 1.0]);

    let e09 = packet.feature(FeatureId::new(b'E', 9).unwrap());
    assert!(e09.row_is_available(0));
    assert_eq!(e09.values, vec![1.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn template_response_rejects_invalid_rational_probe_points() {
    assert!(RationalProbability::new(1, 0).is_err());
    assert!(RationalProbability::new(2, 3).is_ok());
    assert!(RationalProbability::new(4, 3).is_err());
}

#[test]
fn e02_provider_identity_binds_choices_points_and_budgets() {
    let choices = vec![
        TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        TemplateChoice::new(TemplateChoiceKind::Swap, 0, 1),
    ];
    let point_a = vec![vec![
        RationalProbability::new(1, 4).unwrap(),
        RationalProbability::new(3, 4).unwrap(),
    ]];
    let point_b = vec![vec![
        RationalProbability::new(1, 2).unwrap(),
        RationalProbability::new(1, 2).unwrap(),
    ]];
    let mut first = FeatureRegistry::new();
    let mut second = FeatureRegistry::new();
    register_template_response(
        &mut first,
        TemplateResponseConfig::new(choices.clone(), point_a.clone()).unwrap(),
    )
    .unwrap();
    register_template_response(
        &mut second,
        TemplateResponseConfig::new(choices, point_b).unwrap(),
    )
    .unwrap();
    assert_ne!(
        first.specs()[e02_id().index()].producer_version,
        second.specs()[e02_id().index()].producer_version
    );
}

#[test]
fn e02_enforces_its_width_budget_before_producing_a_value() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    let choices = vec![
        TemplateChoice::new(TemplateChoiceKind::Add, 0, 1),
        TemplateChoice::new(TemplateChoiceKind::Add, 1, 0),
    ];
    let points = vec![vec![
        RationalProbability::new(1, 2).unwrap(),
        RationalProbability::new(1, 2).unwrap(),
    ]];
    let config = TemplateResponseConfig::with_budgets(
        choices,
        points,
        TemplateResponseBudgets {
            max_width: 1,
            ..TemplateResponseBudgets::default()
        },
    )
    .unwrap();
    let mut registry = FeatureRegistry::new();
    register_template_response(&mut registry, config).unwrap();
    let mut policy = FeaturePolicy {
        enabled_ids: vec![e02_id()],
        seed: 0,
        schema_id: "template-response-budget-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    policy.symmetry_contract = "augmentation_only".into();
    let packet = registry.build(&[state], &policy).unwrap();
    assert_eq!(
        packet.feature(e02_id()).status,
        vec![FeatureStatus::ResourceLimit]
    );
}

#[test]
fn e03_reports_resource_limit_instead_of_approximating_rational_overflow() {
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 2,
        })
        .unwrap();
    let choices = (0..8)
        .map(|_| TemplateChoice::new(TemplateChoiceKind::Add, 0, 1))
        .collect();
    let points = vec![vec![RationalProbability::new(1, u64::MAX).unwrap(); 8]];
    let config = TemplateResponseConfig::new(choices, points).unwrap();
    let mut registry = FeatureRegistry::new();
    register_template_response(&mut registry, config).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![FeatureId::new(b'E', 3).unwrap()],
        seed: 0,
        schema_id: "template-response-rational-overflow-test-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };
    let packet = registry.build(&[state], &policy).unwrap();
    let e03 = packet.feature(FeatureId::new(b'E', 3).unwrap());
    assert!(!e03.row_is_available(0));
    assert_eq!(e03.status, vec![FeatureStatus::ResourceLimit]);
}
