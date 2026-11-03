/// Explicit physical/experiment contexts needed to register every current model-input provider.
///
/// No contexts are synthesized from matrix width. Each collection must contain
/// enough validated, source-bound records for its provider; registration fails
/// closed when a required collection or resource bound is absent.
pub struct FeatureProviderContexts {
    pub max_n: u32,
    pub labeled_cuts: Vec<crate::rank_features::LabeledCut>,
    pub restricted_intersections: Vec<crate::rank_features::RestrictedIntersection>,
    pub rank_increment_probes: Vec<crate::rank_features::RankIncrementProbe>,
    pub alphabet: crate::alphabet::CnotAlphabet,
    pub selected_residual_gates: Vec<crate::state::CnotGate>,
    pub block_parity_contexts: Vec<crate::block_parities::BlockContext>,
    pub max_parity_patterns: usize,
    pub max_parity_pairs: usize,
    pub matrix_boundaries: Vec<crate::matrix_boundary::MatrixBoundarySpec>,
    pub max_boundary_ports: usize,
    pub max_boundary_equations: usize,
    pub paid_basis_contexts: Vec<crate::paid_basis_features::PaidBasisBlockContext>,
    pub paid_shear_config: crate::paid_basis::PaidShearConfig,
    pub template_response: crate::template_response::TemplateResponseConfig,
    pub bilinear_contexts: Vec<crate::bilinear::BilinearContext>,
    pub shared_relation_contexts: Vec<crate::shared_relation::SharedRelationContext>,
    pub frequency_contexts: Vec<crate::frequency_context::FrequencyContext>,
    pub positive_packet_contexts: Vec<crate::positive_packet::OptionalGatePacket>,
    pub i09_contexts: Vec<crate::i09_quadratic::I09Context>,
    pub i09_limits: crate::i09_quadratic::I09Limits,
    pub i10_config: crate::response::I10Config,
}
