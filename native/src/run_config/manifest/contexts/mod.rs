//! Deserialization and checked construction of explicit feature-provider contexts.

use serde::Deserialize;

mod builder;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AlphabetSpec {
    dimension: u32,
    id: String,
    gates: Vec<GateSpec>,
    costs: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateSpec {
    control: u32,
    target: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FeatureContextsSpec {
    pub(super) max_n: u32,
    labeled_cuts: Vec<LabeledCutSpec>,
    restricted_intersections: Vec<RestrictedIntersectionSpec>,
    rank_increment_probes: Vec<RankIncrementProbeSpec>,
    alphabet: AlphabetSpec,
    selected_residual_gates: Vec<GateSpec>,
    block_parity_contexts: Vec<BlockContextSpec>,
    max_parity_patterns: usize,
    max_parity_pairs: usize,
    matrix_boundaries: Vec<MatrixBoundarySpecInput>,
    max_boundary_ports: usize,
    max_boundary_equations: usize,
    paid_basis_contexts: Vec<PaidBasisBlockContextSpec>,
    paid_shear_config: PaidShearConfigSpec,
    template_response: TemplateResponseSpec,
    bilinear_contexts: Vec<BilinearContextSpec>,
    shared_relation_contexts: Vec<SharedRelationContextSpec>,
    frequency_contexts: Vec<FrequencyContextSpec>,
    positive_packet_contexts: Vec<OptionalGatePacketSpec>,
    i09_contexts: Vec<I09ContextSpec>,
    i09_limits: I09LimitsSpec,
    i10_config: I10ConfigSpec,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LabeledCutSpec {
    n: u32,
    a: Vec<u32>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RestrictedIntersectionSpec {
    n: u32,
    left: Vec<u32>,
    right: Vec<u32>,
    coordinates: Vec<u32>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankIncrementProbeSpec {
    n: u32,
    span_rows: Vec<u32>,
    coordinates: Vec<u32>,
    candidate_support: Vec<u32>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockContextSpec {
    n: u32,
    sources: Vec<u32>,
    targets: Vec<u32>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightSpec {
    numerator: String,
    denominator: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixBoundarySpecInput {
    n: u32,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    protected_inputs: Vec<u32>,
    zero_outputs: Vec<u32>,
    weights: Vec<[WeightSpec; 2]>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PaidBasisBlockContextSpec {
    dimension: u32,
    source_wires: Vec<usize>,
    target_wires: Vec<usize>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PaidShearConfigSpec {
    local_width: usize,
    sweeps: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemplateResponseSpec {
    choices: Vec<TemplateChoiceSpec>,
    points: Vec<Vec<ProbabilitySpec>>,
    budgets: TemplateBudgetsSpec,
}
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChoiceKindSpec {
    Add,
    Swap,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemplateChoiceSpec {
    kind: ChoiceKindSpec,
    control: usize,
    target: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbabilitySpec {
    numerator: u64,
    denominator: u64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemplateBudgetsSpec {
    max_width: usize,
    max_assignments: usize,
    max_evaluations: usize,
    max_bytes: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BilinearContextSpec {
    n: u32,
    rows: Vec<u32>,
    columns: Vec<u32>,
    hidden_width: usize,
    u_rows: Vec<Vec<u64>>,
    v_rows: Vec<Vec<u64>>,
    q: usize,
    a0_rows: Vec<Vec<u64>>,
    d0_rows: Vec<Vec<u64>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedRelationContextSpec {
    n: u32,
    rows: Vec<u32>,
    coordinates: Vec<u32>,
    covectors: Vec<Vec<u64>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrequencyContextSpec {
    n: u32,
    sources: Vec<u32>,
    internal: Vec<u32>,
    targets: Vec<u32>,
    central_rows: Vec<Vec<u64>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionalGatePacketSpec {
    n: u32,
    gates: Vec<[u32; 2]>,
    factors: Vec<LocalBooleanFactorSpec>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalBooleanFactorSpec {
    scope: Vec<u32>,
    allowed: Vec<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct I09ContextSpec {
    n: usize,
    horizon: usize,
    cut: usize,
    prefix_bits: Vec<u8>,
    allowed_gates: Vec<[usize; 2]>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct I09LimitsSpec {
    max_variables: usize,
    max_equations: usize,
    max_bytes: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct I10ConfigSpec {
    max_n: u32,
    max_degree: usize,
    max_layer_states: usize,
    max_transitions: usize,
}
