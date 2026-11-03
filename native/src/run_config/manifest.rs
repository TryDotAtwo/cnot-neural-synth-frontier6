use crate::alphabet::CnotAlphabet;
use crate::features::FeatureProviderContexts;
use crate::matrix_boundary::{MatrixBoundarySpec, RationalWeight};
use crate::model::ModelConfig;
use crate::nn_training::AdamWConfig;
use crate::objective::FullGraphObjectiveConfig;
use crate::{
    BilinearContext, BlockContext, CnotGate, FrequencyContext, I09Context, I09Limits, I10Config,
    LabeledCut, LocalBooleanFactor, OptionalGatePacket, PaidBasisBlockContext, PaidShearConfig,
    RankIncrementProbe, RationalProbability, RestrictedIntersection, SharedRelationContext,
    TemplateChoice, TemplateChoiceKind, TemplateResponseBudgets, TemplateResponseConfig,
};
use num_bigint::BigUint;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

const RUN_SCHEMA: &str = "cnot103_training_run_v1";

#[derive(Debug)]
pub struct RunManifest {
    pub(super) dataset: PinnedArtifact,
    pub(super) exact_catalogs: PinnedArtifact,
    pub(super) checkpoint_dir: PathBuf,
    pub(super) resume: bool,
    pub(super) training: TrainingSpec,
    pub(super) feature_policy: FeaturePolicySpec,
    pub(super) model: ModelSpec,
    pub(super) optimizer: AdamWConfig,
    pub(super) objective: ObjectiveSpec,
    pub(super) alphabets: Vec<AlphabetSpec>,
    pub(super) feature_contexts: FeatureContextsSpec,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRunManifest {
    schema: String,
    dataset: PinnedArtifact,
    exact_catalogs: PinnedArtifact,
    checkpoint_dir: PathBuf,
    resume: bool,
    training: TrainingSpec,
    feature_policy: FeaturePolicySpec,
    model: ModelSpec,
    optimizer: AdamWConfig,
    objective: ObjectiveSpec,
    alphabets: Vec<AlphabetSpec>,
    feature_contexts: FeatureContextsSpec,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PinnedArtifact {
    pub(super) path: PathBuf,
    pub(super) sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrainingSpec {
    pub(super) seed: u64,
    pub(super) cuda_device: usize,
    pub(super) attempts: u64,
    pub(super) batch_size: usize,
    pub(super) checkpoint_every_attempts: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FeaturePolicySpec {
    pub(super) schema_id: String,
    pub(super) solver_assisted: bool,
    pub(super) evaluations: ExplicitNullable<usize>,
    pub(super) max_bytes: ExplicitNullable<usize>,
}

#[derive(Debug)]
pub(super) struct ExplicitNullable<T>(Option<T>);

impl<T> ExplicitNullable<T> {
    pub(super) fn into_option(self) -> Option<T> {
        self.0
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ExplicitNullable<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelSpec {
    wire_dim: i64,
    pair_dim: i64,
    structural_blocks: usize,
    attention_heads: i64,
    feedforward_dim: i64,
    fusion_blocks: usize,
    basis_proposal_hidden_dim: i64,
}

impl ModelSpec {
    pub(super) fn build(self) -> ModelConfig {
        ModelConfig {
            wire_dim: self.wire_dim,
            pair_dim: self.pair_dim,
            structural_blocks: self.structural_blocks,
            attention_heads: self.attention_heads,
            feedforward_dim: self.feedforward_dim,
            fusion_blocks: self.fusion_blocks,
            basis_proposal_hidden_dim: self.basis_proposal_hidden_dim,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObjectiveSpec {
    bellman_weight: f64,
    supervision_weight: f64,
    qhat_weight: f64,
    teacher_weight: f64,
    neighbor_chunk_size: usize,
    teacher_temperature: f64,
}

impl ObjectiveSpec {
    pub(super) fn build(self) -> FullGraphObjectiveConfig {
        FullGraphObjectiveConfig {
            bellman_weight: self.bellman_weight,
            supervision_weight: self.supervision_weight,
            qhat_weight: self.qhat_weight,
            teacher_weight: self.teacher_weight,
            neighbor_chunk_size: self.neighbor_chunk_size,
            teacher_temperature: self.teacher_temperature,
            external_admission: false,
            external_admission_sha256: None,
            require_all_model_inputs: true,
            compact_exposures: false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AlphabetSpec {
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

impl GateSpec {
    pub(super) fn build(self) -> CnotGate {
        CnotGate {
            control: self.control,
            target: self.target,
        }
    }
}

impl AlphabetSpec {
    pub(super) fn build(&self) -> Result<CnotAlphabet, String> {
        CnotAlphabet::new(
            self.dimension,
            self.id.clone(),
            self.gates.iter().copied().map(GateSpec::build).collect(),
            self.costs.clone(),
        )
        .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FeatureContextsSpec {
    max_n: u32,
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

impl FeatureContextsSpec {
    pub(super) fn build(self) -> Result<FeatureProviderContexts, String> {
        let alphabet = self.alphabet.build()?;
        let labeled_cuts = self
            .labeled_cuts
            .into_iter()
            .map(|x| LabeledCut::new(x.n, x.a).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let restricted_intersections = self
            .restricted_intersections
            .into_iter()
            .map(|x| {
                RestrictedIntersection::new(x.n, x.left, x.right, x.coordinates)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rank_increment_probes = self
            .rank_increment_probes
            .into_iter()
            .map(|x| {
                RankIncrementProbe::new(x.n, x.span_rows, x.coordinates, x.candidate_support)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let block_parity_contexts = self
            .block_parity_contexts
            .into_iter()
            .map(|x| BlockContext::new(x.n, x.sources, x.targets).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let matrix_boundaries = self
            .matrix_boundaries
            .into_iter()
            .map(MatrixBoundarySpecInput::build)
            .collect::<Result<Vec<_>, _>>()?;
        let paid_basis_contexts = self
            .paid_basis_contexts
            .into_iter()
            .map(|x| {
                PaidBasisBlockContext::new(x.dimension, x.source_wires, x.target_wires)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let template_response = self.template_response.build()?;
        let bilinear_contexts = self
            .bilinear_contexts
            .into_iter()
            .map(|x| {
                BilinearContext::new(
                    x.n,
                    x.rows,
                    x.columns,
                    x.hidden_width,
                    x.u_rows,
                    x.v_rows,
                    x.q,
                    x.a0_rows,
                    x.d0_rows,
                )
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_relation_contexts = self
            .shared_relation_contexts
            .into_iter()
            .map(|x| {
                SharedRelationContext::new(x.n, x.rows, x.coordinates, x.covectors)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let frequency_contexts = self
            .frequency_contexts
            .into_iter()
            .map(|x| {
                FrequencyContext::new(x.n, x.sources, x.internal, x.targets, x.central_rows)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let positive_packet_contexts = self
            .positive_packet_contexts
            .into_iter()
            .map(OptionalGatePacketSpec::build)
            .collect::<Result<Vec<_>, _>>()?;
        let i09_contexts = self
            .i09_contexts
            .into_iter()
            .map(|x| {
                I09Context::new(
                    x.n,
                    x.horizon,
                    x.cut,
                    x.prefix_bits,
                    x.allowed_gates.into_iter().map(|[c, t]| (c, t)).collect(),
                )
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FeatureProviderContexts {
            max_n: self.max_n,
            labeled_cuts,
            restricted_intersections,
            rank_increment_probes,
            alphabet,
            selected_residual_gates: self
                .selected_residual_gates
                .into_iter()
                .map(GateSpec::build)
                .collect(),
            block_parity_contexts,
            max_parity_patterns: self.max_parity_patterns,
            max_parity_pairs: self.max_parity_pairs,
            matrix_boundaries,
            max_boundary_ports: self.max_boundary_ports,
            max_boundary_equations: self.max_boundary_equations,
            paid_basis_contexts,
            paid_shear_config: PaidShearConfig {
                local_width: self.paid_shear_config.local_width,
                sweeps: self.paid_shear_config.sweeps,
            },
            template_response,
            bilinear_contexts,
            shared_relation_contexts,
            frequency_contexts,
            positive_packet_contexts,
            i09_contexts,
            i09_limits: I09Limits {
                max_variables: self.i09_limits.max_variables,
                max_equations: self.i09_limits.max_equations,
                max_bytes: self.i09_limits.max_bytes,
            },
            i10_config: I10Config {
                max_n: self.i10_config.max_n,
                max_degree: self.i10_config.max_degree,
                max_layer_states: self.i10_config.max_layer_states,
                max_transitions: self.i10_config.max_transitions,
            },
        })
    }
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
impl MatrixBoundarySpecInput {
    pub(super) fn build(self) -> Result<MatrixBoundarySpec, String> {
        let mut spec = MatrixBoundarySpec::new(self.n, self.inputs, self.outputs)
            .map_err(|e| e.to_string())?
            .with_protected_inputs(self.protected_inputs)
            .map_err(|e| e.to_string())?
            .with_zero_outputs(self.zero_outputs)
            .map_err(|e| e.to_string())?;
        if !self.weights.is_empty() {
            let weights = self
                .weights
                .into_iter()
                .map(|[first, second]| {
                    Ok((
                        RationalWeight::new(first.numerator, first.denominator)
                            .map_err(|e| e.to_string())?,
                        RationalWeight::new(second.numerator, second.denominator)
                            .map_err(|e| e.to_string())?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            spec = spec.with_weights(weights).map_err(|e| e.to_string())?;
        }
        Ok(spec)
    }
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
impl TemplateResponseSpec {
    pub(super) fn build(self) -> Result<TemplateResponseConfig, String> {
        let choices = self
            .choices
            .into_iter()
            .map(|x| TemplateChoice::new(x.kind.into(), x.control, x.target))
            .collect();
        let points = self
            .points
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|x| {
                        RationalProbability::new(x.numerator, x.denominator)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        TemplateResponseConfig::with_budgets(
            choices,
            points,
            TemplateResponseBudgets {
                max_width: self.budgets.max_width,
                max_assignments: self.budgets.max_assignments,
                max_evaluations: self.budgets.max_evaluations,
                max_bytes: self.budgets.max_bytes,
            },
        )
        .map_err(|e| e.to_string())
    }
}
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChoiceKindSpec {
    Add,
    Swap,
}
impl From<ChoiceKindSpec> for TemplateChoiceKind {
    fn from(x: ChoiceKindSpec) -> Self {
        match x {
            ChoiceKindSpec::Add => Self::Add,
            ChoiceKindSpec::Swap => Self::Swap,
        }
    }
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
impl OptionalGatePacketSpec {
    pub(super) fn build(self) -> Result<OptionalGatePacket, String> {
        let gates = self.gates.into_iter().map(|[c, t]| (c, t)).collect();
        let factors = self
            .factors
            .into_iter()
            .map(|x| {
                let allowed = x
                    .allowed
                    .iter()
                    .map(|s| {
                        s.parse::<BigUint>()
                            .map_err(|e| format!("invalid decimal BigUint: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                LocalBooleanFactor::new(x.scope, allowed).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        OptionalGatePacket::new(self.n, gates, factors).map_err(|e| e.to_string())
    }
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

pub fn parse_run_manifest(raw: &[u8]) -> Result<RunManifest, String> {
    let value: Value = serde_json::from_slice(raw).map_err(|e| format!("invalid run JSON: {e}"))?;
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| "run manifest requires a string schema".to_owned())?;
    if schema != RUN_SCHEMA {
        return Err(format!(
            "run manifest schema must be {RUN_SCHEMA}, got {schema}"
        ));
    }
    let raw: RawRunManifest =
        serde_json::from_value(value).map_err(|e| format!("invalid run manifest: {e}"))?;
    if raw.schema != RUN_SCHEMA {
        return Err(format!("run manifest schema must be {RUN_SCHEMA}"));
    }
    let manifest = RunManifest {
        dataset: raw.dataset,
        exact_catalogs: raw.exact_catalogs,
        checkpoint_dir: raw.checkpoint_dir,
        resume: raw.resume,
        training: raw.training,
        feature_policy: raw.feature_policy,
        model: raw.model,
        optimizer: raw.optimizer,
        objective: raw.objective,
        alphabets: raw.alphabets,
        feature_contexts: raw.feature_contexts,
    };
    manifest.validate()?;
    Ok(manifest)
}

impl RunManifest {
    fn validate(&self) -> Result<(), String> {
        validate_sha256(&self.dataset.sha256, "dataset")?;
        validate_sha256(&self.exact_catalogs.sha256, "exact catalogs")?;
        if self.checkpoint_dir.as_os_str().is_empty() {
            return Err("checkpoint_dir must not be empty".into());
        }
        if self.training.attempts == 0
            || self.training.batch_size == 0
            || self.training.checkpoint_every_attempts == 0
        {
            return Err(
                "attempts, batch_size, and checkpoint_every_attempts must be positive".into(),
            );
        }
        if self.feature_policy.schema_id.trim().is_empty() {
            return Err("feature_policy.schema_id must be nonempty".into());
        }
        if self.alphabets.is_empty() {
            return Err("at least one exact alphabet must be provided".into());
        }
        if self.feature_contexts.max_n < 2 {
            return Err("feature_contexts.max_n must be at least two".into());
        }
        let w = [
            self.objective.bellman_weight,
            self.objective.supervision_weight,
            self.objective.qhat_weight,
            self.objective.teacher_weight,
        ];
        if w.iter().any(|x| !x.is_finite() || *x < 0.0)
            || w.iter().sum::<f64>() <= 0.0
            || self.objective.neighbor_chunk_size == 0
            || !self.objective.teacher_temperature.is_finite()
            || self.objective.teacher_temperature <= 0.0
        {
            return Err("objective weights and budgets are invalid".into());
        }
        if self.feature_policy.evaluations.0 == Some(0)
            || self.feature_policy.max_bytes.0 == Some(0)
        {
            return Err("feature evaluation and byte limits, when set, must be positive".into());
        }
        Ok(())
    }
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!(
            "{label} SHA-256 must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FeaturePolicy, FeatureRegistry};
    use serde_json::json;

    #[test]
    fn rejects_wrong_schema_before_artifact_io() {
        let error = parse_run_manifest(br#"{"schema":"wrong"}"#).unwrap_err();
        assert!(error.contains("schema"));
    }
    #[test]
    fn rejects_unknown_top_level_fields() {
        let error =
            parse_run_manifest(br#"{"schema":"cnot103_training_run_v1","invented_default":true}"#)
                .unwrap_err();
        assert!(error.contains("unknown field"));
    }
    fn explicit_test_contexts_json() -> Value {
        json!({
            "max_n": 3,
            "labeled_cuts": [{"n": 3, "a": [0]}],
            "restricted_intersections": [{"n": 3, "left": [0], "right": [1], "coordinates": [0]}],
            "rank_increment_probes": [{"n": 3, "span_rows": [0], "coordinates": [0], "candidate_support": []}],
            "alphabet": {
                "dimension": 3,
                "id": "all_to_all_unit_cnot:3",
                "gates": [
                    {"control":0,"target":1},{"control":0,"target":2},
                    {"control":1,"target":0},{"control":1,"target":2},
                    {"control":2,"target":0},{"control":2,"target":1}
                ],
                "costs": [1,1,1,1,1,1]
            },
            "selected_residual_gates": [{"control":0,"target":1}],
            "block_parity_contexts": [{"n":3,"sources":[0],"targets":[1]}],
            "max_parity_patterns": 1024,
            "max_parity_pairs": 4096,
            "matrix_boundaries": [{
                "n":3,"inputs":[0],"outputs":[2],"protected_inputs":[],
                "zero_outputs":[],"weights":[]
            }],
            "max_boundary_ports": 8,
            "max_boundary_equations": 1024,
            "paid_basis_contexts": [{"dimension":3,"source_wires":[0],"target_wires":[1]}],
            "paid_shear_config": {"local_width":2,"sweeps":1},
            "template_response": {
                "choices":[
                    {"kind":"add","control":0,"target":1},
                    {"kind":"swap","control":1,"target":2}
                ],
                "points":[[
                    {"numerator":1,"denominator":2},
                    {"numerator":1,"denominator":2}
                ]],
                "budgets":{
                    "max_width":3,"max_assignments":16,
                    "max_evaluations":128,"max_bytes":65536
                }
            },
            "bilinear_contexts":[{
                "n":3,"rows":[0],"columns":[1],"hidden_width":1,
                "u_rows":[[1]],"v_rows":[[1]],"q":1,
                "a0_rows":[[0]],"d0_rows":[[0]]
            }],
            "shared_relation_contexts":[{
                "n":3,"rows":[0],"coordinates":[1],"covectors":[[1]]
            }],
            "frequency_contexts":[{
                "n":3,"sources":[0],"internal":[1],"targets":[2],
                "central_rows":[[1]]
            }],
            "positive_packet_contexts":[{
                "n":3,"gates":[[0,1]],
                "factors":[{"scope":[0],"allowed":["0","1"]}]
            }],
            "i09_contexts":[{
                "n":3,"horizon":1,"cut":0,"prefix_bits":[],
                "allowed_gates":[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
            }],
            "i09_limits":{"max_variables":1024,"max_equations":4096,"max_bytes":1048576},
            "i10_config":{"max_n":3,"max_degree":1,"max_layer_states":4096,"max_transitions":65536}
        })
    }

    #[test]
    fn explicit_context_bundle_registers_all_model_input_ids() {
        let contexts: FeatureContextsSpec = serde_json::from_value(explicit_test_contexts_json())
            .expect("complete explicit test profile must parse");
        let registry = FeatureRegistry::with_all_model_input_providers(
            contexts
                .build()
                .expect("context constructors must validate"),
        )
        .expect("complete profile providers must register");
        let mut policy = FeaturePolicy::all_model_inputs_augmentation_only("test-profile");
        policy.solver_assisted = true;
        assert_eq!(
            registry
                .selected_specs(&policy)
                .expect("all inputs selected")
                .len(),
            crate::schema::MODEL_INPUT_COUNT
        );
    }

    #[test]
    fn complete_manifest_parses_with_explicit_null_limits() {
        let contexts = explicit_test_contexts_json();
        let alphabet = contexts["alphabet"].clone();
        let value = json!({
            "schema": RUN_SCHEMA,
            "dataset": {"path":"frozen.json","sha256":"a".repeat(64)},
            "exact_catalogs": {"path":"catalogs.json","sha256":"b".repeat(64)},
            "checkpoint_dir":"runs/test",
            "resume":false,
            "training":{
                "seed":42,"cuda_device":0,"attempts":1,
                "batch_size":1,"checkpoint_every_attempts":1
            },
            "feature_policy":{
                "schema_id":"runtime_packet_layout_v24",
                "solver_assisted":true,"evaluations":null,"max_bytes":null
            },
            "model":{
                "wire_dim":8,"pair_dim":4,"structural_blocks":2,
                "attention_heads":2,"feedforward_dim":16,
                "fusion_blocks":1,"basis_proposal_hidden_dim":8
            },
            "optimizer":{
                "learning_rate":0.0001,"beta1":0.9,"beta2":0.999,
                "epsilon":0.00000001,"weight_decay":0.01
            },
            "objective":{
                "bellman_weight":1.0,"supervision_weight":1.0,
                "qhat_weight":0.1,"teacher_weight":0.0,
                "neighbor_chunk_size":8,"teacher_temperature":4.0
            },
            "alphabets":[alphabet],
            "feature_contexts":contexts
        });
        let parsed = parse_run_manifest(&serde_json::to_vec(&value).unwrap())
            .expect("complete manifest must parse without opening its artifacts");
        assert_eq!(parsed.training.attempts, 1);
        assert!(parsed.feature_policy.evaluations.0.is_none());
        assert!(parsed.feature_policy.max_bytes.0.is_none());
    }

    #[test]
    fn sha_pins_are_lowercase_and_exactly_256_bits() {
        assert!(validate_sha256(&"a".repeat(64), "test").is_ok());
        assert!(validate_sha256(&"A".repeat(64), "test").is_err());
        assert!(validate_sha256(&"a".repeat(63), "test").is_err());
    }
}
