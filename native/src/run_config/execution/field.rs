use super::artifacts::PinnedRunData;
use super::feature_runtime;
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::model::FusionDistanceField;
use crate::objective::FullGraphObjective;
use crate::run_config::manifest::{
    FeatureContextsSpec, FeaturePolicySpec, ModelSpec, ObjectiveSpec, TrainingSpec,
};
use crate::{FeatureRegistry, FrozenDataset};
use std::collections::{BTreeSet, HashMap};
use tch::{nn::VarStore, Device};

/// Reconstruct the physical providers, exact objective and model from the same
/// artifacts for training and for read-only checkpoint evaluation.
pub(super) struct BoundField {
    pub dataset: FrozenDataset,
    pub registry: FeatureRegistry,
    pub plans: HashMap<u32, DeviceFeaturePlan>,
    pub selected_count: usize,
    pub objective: FullGraphObjective,
    pub model: FusionDistanceField,
    pub variables: VarStore,
    pub replica: Option<BoundReplica>,
}

pub(super) struct BoundReplica {
    pub registry: FeatureRegistry,
    pub plans: HashMap<u32, DeviceFeaturePlan>,
    pub model: FusionDistanceField,
    pub variables: VarStore,
}
impl BoundField {
    pub(super) fn build(
        data: PinnedRunData,
        contexts: FeatureContextsSpec,
        policy_spec: FeaturePolicySpec,
        model: ModelSpec,
        objective: ObjectiveSpec,
        training: &TrainingSpec,
        widths: BTreeSet<u32>,
        replica_device: Option<usize>,
    ) -> Result<Self, String> {
        let PinnedRunData {
            alphabets,
            catalogs,
            dataset,
        } = data;
        let feature_runtime::FeatureRuntime {
            registry,
            policy: master_policy,
            selected_count,
            plans,
        } = feature_runtime::build(
            contexts.clone(),
            policy_spec.clone(),
            &alphabets,
            training.seed,
            widths.clone(),
            training.cuda_device,
        )?;
        let replica_runtime = replica_device
            .map(|index| {
                feature_runtime::build(
                    contexts,
                    policy_spec.clone(),
                    &alphabets,
                    training.seed,
                    widths,
                    index,
                )
            })
            .transpose()?;
        if let Some(replica) = &replica_runtime {
            if replica.selected_count != selected_count
                || replica.registry.contract_sha256() != registry.contract_sha256()
                || replica.policy.id().map_err(|e| e.to_string())?
                    != master_policy.id().map_err(|e| e.to_string())?
            {
                return Err("replica feature runtime differs from the master contract".into());
            }
        }
        let objective = FullGraphObjective::new(
            alphabets,
            &registry,
            master_policy.clone(),
            catalogs.into_values().collect(),
            dataset.forbidden_keys.iter().cloned().collect(),
            objective.build(),
        )
        .map_err(|e| format!("objective construction failed: {e}"))?;
        tch::manual_seed(training.seed as i64);
        let device = Device::Cuda(training.cuda_device);
        let variables = VarStore::new(device);
        let model_config = model.build();
        let model = FusionDistanceField::new_for_policy(
            &variables.root(),
            &registry,
            &master_policy,
            device,
            model_config.clone(),
        )
        .map_err(|e| e.to_string())?;
        let replica = if let (Some(index), Some(runtime)) = (replica_device, replica_runtime) {
            let variables = VarStore::new(Device::Cuda(index));
            let model = FusionDistanceField::new_for_policy(
                &variables.root(),
                &runtime.registry,
                &runtime.policy,
                Device::Cuda(index),
                model_config,
            )
            .map_err(|e| e.to_string())?;
            Some(BoundReplica {
                registry: runtime.registry,
                plans: runtime.plans,
                model,
                variables,
            })
        } else {
            None
        };
        Ok(Self {
            registry,
            dataset,
            plans,
            selected_count,
            objective,
            model,
            variables,
            replica,
        })
    }
}
