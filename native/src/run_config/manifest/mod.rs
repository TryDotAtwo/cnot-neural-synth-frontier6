use crate::model::ModelConfig;
use crate::nn_training::AdamWConfig;
use crate::objective::FullGraphObjectiveConfig;
use crate::features::{FeaturePolicy, FeatureRegistry};
use serde::Deserialize;
use std::path::PathBuf;

mod contexts;
pub(super) use contexts::{AlphabetSpec, FeatureContextsSpec};

const RUN_SCHEMA: &str = "cnot103_training_run_v1";

#[derive(Debug)]
pub struct RunManifest {
    pub(super) dataset: PinnedArtifact,
    pub(super) exact_catalogs: Option<PinnedArtifact>,
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
    exact_catalogs: Option<PinnedArtifact>,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrainingSpec {
    pub(super) seed: u64,
    pub(super) cuda_device: usize,
    #[serde(default)]
    pub(super) cuda_devices: Vec<usize>,
    pub(super) attempts: u64,
    #[serde(default)]
    pub(super) target_attempts: Option<u64>,
    pub(super) batch_size: usize,
    pub(super) checkpoint_every_attempts: u64,
}

impl RunManifest {
    /// Validate provider construction and the complete runtime model-input selection
    /// without opening the pinned dataset or touching a CUDA device.
    pub fn validate_feature_profile(&self) -> Result<usize, String> {
        let f08_max_n = self.feature_contexts.f08_max_n();
        let f08_max_changed_rows = self.feature_contexts.f08_max_changed_rows();
        let f08_max_component_events = self.feature_contexts.f08_max_component_events();
        let f08_max_history_counts = self.feature_contexts.f08_max_history_counts();
        let contexts = self.feature_contexts.clone().build()?;
        let registry = FeatureRegistry::with_all_model_input_providers_f08_limit(
            contexts,
            f08_max_n,
            f08_max_changed_rows,
            f08_max_component_events,
            f08_max_history_counts,
        )
        .map_err(|error| error.to_string())?;
        let mut policy = FeaturePolicy::all_model_inputs_augmentation_only(
            self.feature_policy.schema_id.clone(),
        );
        policy.seed = self.training.seed;
        policy.solver_assisted = self.feature_policy.solver_assisted;
        policy.evaluations = self.feature_policy.evaluations.0;
        policy.max_bytes = self.feature_policy.max_bytes.0;
        let selected = registry
            .selected_specs(&policy)
            .map_err(|error| error.to_string())?;
        let expected = crate::schema::MODEL_INPUT_COUNT;
        if selected.len() != expected {
            return Err(format!(
                "feature profile selects {} model inputs, expected {expected}",
                selected.len()
            ));
        }
        Ok(selected.len())
    }
}

impl TrainingSpec {
    pub(super) fn effective_cuda_devices(&self) -> Vec<usize> {
        if self.cuda_devices.is_empty() {
            vec![self.cuda_device]
        } else {
            self.cuda_devices.clone()
        }
    }

    /// Cap this invocation at the remaining absolute target, when configured.
    pub(super) fn attempts_for_progress(&self, attempted: u64) -> Result<u64, String> {
        match self.target_attempts {
            Some(target) => {
                let remaining = target.checked_sub(attempted)
                    .ok_or_else(|| "checkpoint progress exceeds target attempt count".to_owned())?;
                Ok(remaining.min(self.attempts))
            }
            None => Ok(self.attempts),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FeaturePolicySpec {
    #[serde(default = "default_feature_schema")]
    pub(super) schema_id: String,
    pub(super) solver_assisted: bool,
    pub(super) evaluations: ExplicitNullable<usize>,
    pub(super) max_bytes: ExplicitNullable<usize>,
}

fn default_feature_schema() -> String {
    crate::schema::MISSING_TOKEN_LAYOUT_ID.to_owned()
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone, Deserialize)]
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
    pub(super) fn build(&self) -> ModelConfig {
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
    #[serde(default)]
    sampled_neighbor_count: Option<usize>,
    #[serde(default)]
    adaptive_neighbor_chunk: bool,
    #[serde(default)]
    share_candidate_wire_projections: bool,
    #[serde(default)]
    share_training_wire_projections_min_width: Option<u32>,
    #[serde(default)]
    cache_root_f08: bool,
    #[serde(default)]
    sequential_batch_gradients: bool,
    #[serde(default)]
    prevalidated_device_closure_audit: bool,
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
            sampled_neighbor_count: self.sampled_neighbor_count,
            adaptive_neighbor_chunk: self.adaptive_neighbor_chunk,
            share_candidate_wire_projections: self.share_candidate_wire_projections,
            share_training_wire_projections_min_width: self.share_training_wire_projections_min_width,
            cache_root_f08: self.cache_root_f08,
            sequential_batch_gradients: self.sequential_batch_gradients,
            teacher_temperature: self.teacher_temperature,
            external_admission: false,
            external_admission_sha256: None,
            require_all_model_inputs: true,
            compact_exposures: self.prevalidated_device_closure_audit,
            prevalidated_device_closure: self.prevalidated_device_closure_audit,
        }
    }

    pub(super) fn runtime_id(&self) -> &'static str {
        if self.sampled_neighbor_count.is_some() {
            "cuda_qhat_sampled_bellman_v1"
        } else {
            "cuda_primitive_full_graph_v1"
        }
    }
}

mod parser;
mod validation;

#[cfg(test)]
mod tests;

pub use parser::parse_run_manifest;
