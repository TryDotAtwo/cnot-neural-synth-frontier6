use crate::model::ModelConfig;
use crate::nn_training::AdamWConfig;
use crate::objective::FullGraphObjectiveConfig;
use serde::Deserialize;
use std::path::PathBuf;

mod contexts;
pub(super) use contexts::{AlphabetSpec, FeatureContextsSpec};

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

mod parser;
mod validation;

#[cfg(test)]
mod tests;

pub use parser::parse_run_manifest;
