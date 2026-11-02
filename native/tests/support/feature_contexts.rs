use cnot103_native::{
    BilinearContext, BlockContext, CnotAlphabet, CnotGate, FeatureProviderContexts,
    FrequencyContext, I09Context, I09Limits, I10Config, LabeledCut, MatrixBoundarySpec,
    OptionalGatePacket, PaidBasisBlockContext, PaidShearConfig, RankIncrementProbe,
    RationalProbability, RestrictedIntersection, SharedRelationContext, TemplateChoice,
    TemplateChoiceKind, TemplateResponseConfig,
};

pub fn complete_model_input_contexts() -> FeatureProviderContexts {
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
