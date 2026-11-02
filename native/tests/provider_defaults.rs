use cnot103_native::{FeatureId, FeaturePolicy, FeatureRegistry, FEATURE_COUNT};
use std::collections::HashSet;

fn id(family: u8, ordinal: u8) -> FeatureId {
    FeatureId::new(family, ordinal).expect("feature ID is canonical")
}

#[test]
fn context_free_provider_profile_registers_exactly_64_ids_and_keeps_the_rest_open() {
    let registry = FeatureRegistry::with_default_context_free_providers(8)
        .expect("bounded context-free profile should build");

    let missing = [
        (b'C', 2),
        (b'C', 3),
        (b'C', 4),
        (b'C', 6),
        (b'C', 7),
        (b'C', 8),
        (b'C', 9),
        (b'C', 10),
        (b'D', 1),
        (b'D', 2),
        (b'D', 4),
        (b'D', 5),
        (b'D', 6),
        (b'D', 7),
        (b'D', 8),
        (b'D', 9),
        (b'D', 10),
        (b'D', 11),
        (b'D', 12),
        (b'D', 13),
        (b'D', 14),
        (b'E', 2),
        (b'E', 3),
        (b'E', 6),
        (b'E', 7),
        (b'E', 8),
        (b'E', 9),
        (b'H', 7),
        (b'I', 1),
        (b'I', 2),
        (b'I', 3),
        (b'I', 4),
        (b'I', 5),
        (b'I', 6),
        (b'I', 7),
        (b'I', 8),
        (b'I', 9),
        (b'I', 10),
        (b'I', 11),
    ];
    let missing = missing
        .into_iter()
        .map(|(family, ordinal)| id(family, ordinal))
        .collect::<HashSet<_>>();
    let mut available_policy =
        FeaturePolicy::complete_103_augmentation_only("context-free-provider-profile-v1");
    available_policy.solver_assisted = true;
    available_policy.enabled_ids = (0..FEATURE_COUNT)
        .map(|index| FeatureId::from_index(index).unwrap())
        .filter(|feature_id| !missing.contains(feature_id))
        .collect();
    let selected = registry.selected_specs(&available_policy).unwrap();
    assert_eq!(selected.len(), 64);
    assert_eq!(
        selected.iter().map(|spec| spec.id).collect::<Vec<_>>(),
        available_policy.enabled_ids
    );

    let mut full_policy =
        FeaturePolicy::complete_103_augmentation_only("context-free-provider-profile-v1");
    full_policy.solver_assisted = true;
    let error = registry
        .selected_specs(&full_policy)
        .unwrap_err()
        .to_string();
    assert_eq!(
        error.matches("no executable native provider").count(),
        missing.len(),
        "{error}"
    );
    for feature_id in missing {
        assert!(error.contains(&feature_id.to_string()), "{error}");
    }
}

#[test]
fn context_free_provider_profile_rejects_invalid_guarded_width() {
    let error = match FeatureRegistry::with_default_context_free_providers(1) {
        Ok(_) => panic!("width below two must be rejected"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("at least two"), "{error}");
}

use cnot103_native::{
    BilinearContext, BlockContext, CnotAlphabet, CnotGate, FeatureProviderContexts,
    FrequencyContext, I09Context, I09Limits, I10Config, LabeledCut, MatrixBoundarySpec,
    OptionalGatePacket, PaidBasisBlockContext, PaidShearConfig, RankIncrementProbe,
    RationalProbability, RestrictedIntersection, SharedRelationContext, TemplateChoice,
    TemplateChoiceKind, TemplateResponseConfig,
};

fn complete_model_input_contexts() -> FeatureProviderContexts {
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

    FeatureProviderContexts {
        max_n: 8,
        labeled_cuts: vec![LabeledCut::new(3, vec![0]).unwrap()],
        restricted_intersections: vec![
            RestrictedIntersection::new(3, vec![0], vec![1], vec![2]).unwrap()
        ],
        rank_increment_probes: vec![RankIncrementProbe::new(3, vec![0], vec![1], vec![2]).unwrap()],
        alphabet: CnotAlphabet::all_to_all_unit(3).unwrap(),
        selected_residual_gates: vec![CnotGate {
            control: 0,
            target: 1,
        }],
        block_parity_contexts: vec![BlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap()],
        max_parity_patterns: 8,
        max_parity_pairs: 8,
        matrix_boundaries: vec![MatrixBoundarySpec::new(3, vec![0, 1], vec![2]).unwrap()],
        max_boundary_ports: 512,
        max_boundary_equations: 512,
        paid_basis_contexts: vec![PaidBasisBlockContext::new(
            10,
            (0..4).collect(),
            (4..10).collect(),
        )
        .unwrap()],
        paid_shear_config: PaidShearConfig::source_exact(),
        template_response: TemplateResponseConfig::new(choices, points).unwrap(),
        bilinear_contexts: vec![BilinearContext::new(
            2,
            vec![1],
            vec![0],
            2,
            vec![vec![1]],
            vec![vec![1], vec![0]],
            1,
            vec![vec![1]],
            vec![vec![0]],
        )
        .unwrap()],
        shared_relation_contexts: vec![SharedRelationContext::new(
            3,
            vec![0, 1],
            vec![0, 1],
            vec![vec![1]],
        )
        .unwrap()],
        frequency_contexts: vec![FrequencyContext::new(
            3,
            vec![0],
            vec![1],
            vec![2],
            vec![vec![1]],
        )
        .unwrap()],
        positive_packet_contexts: vec![OptionalGatePacket::new(3, vec![(0, 2)], vec![]).unwrap()],
        i09_contexts: vec![I09Context::new(2, 1, 3, vec![0, 0, 0], vec![(0, 1), (1, 0)]).unwrap()],
        i09_limits: I09Limits::default(),
        i10_config: I10Config {
            max_n: 2,
            max_degree: 2,
            max_layer_states: 32,
            max_transitions: 128,
        },
    }
}

#[test]
fn full_model_input_profile_registers_102_features_and_leaves_only_h07_diagnostic() {
    let registry = FeatureRegistry::with_all_model_input_providers(complete_model_input_contexts())
        .expect("all required typed contexts should register");

    let mut available_policy =
        FeaturePolicy::complete_103_augmentation_only("all-model-input-profile-v1");
    available_policy.solver_assisted = true;
    available_policy
        .enabled_ids
        .retain(|feature_id| *feature_id != id(b'H', 7));
    let selected = registry.selected_specs(&available_policy).unwrap();
    assert_eq!(selected.len(), 102);
    assert_eq!(
        selected.iter().map(|spec| spec.id).collect::<Vec<_>>(),
        available_policy.enabled_ids
    );

    let mut complete_policy =
        FeaturePolicy::complete_103_augmentation_only("all-model-input-profile-v1");
    complete_policy.solver_assisted = true;
    let error = registry
        .selected_specs(&complete_policy)
        .unwrap_err()
        .to_string();
    assert_eq!(
        error.matches("no executable native provider").count(),
        1,
        "{error}"
    );
    assert!(
        error.contains("H07: no executable native provider"),
        "{error}"
    );
}
