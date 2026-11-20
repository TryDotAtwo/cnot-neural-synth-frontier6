//! Setup-time binding of the implemented CUDA producers.
//!
//! The plan binds all 102 model inputs against their registered source contracts.
//! Training and post-forward diagnostics have separate entry points.
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeaturePacket, DeviceFeatureStatus};
use crate::{FeatureId, FeaturePolicy, FeatureRegistry, FeatureSpec};
use tch::{Device, Kind, Tensor};

mod basis_gradient;
mod block_histogram;
mod boundary;
mod budget;
mod column_diffusion;
mod contexts;
mod contextual_values;
mod pair_diffusion;
mod shared_word;
mod stage26;
use contexts::PreparedContexts;
pub use contexts::{
    BlockParityPlanInputs, DeviceFeatureContextInputs, MatrixBoundaryPlanInputs,
    PaidBasisPlanInputs, SelectedResidualPlanInputs,
};
pub use stage26::Stage26PlanInputs;
#[derive(Clone, Copy, Debug)]
pub struct GuardedPlanInputs {
    pub max_n: u32,
    pub max_entries: usize,
}
mod frequency;
pub use frequency::FrequencyPlanInputs;
mod shared_relation;
pub use shared_relation::SharedRelationPlanInputs;
mod bilinear;
pub use bilinear::BilinearPlanInputs;
mod positive_packet;
pub use positive_packet::PositivePacketPlanInputs;

#[derive(Clone, Copy)]
pub struct QuadraticPlanInputs<'a> {
    pub contexts: &'a [crate::I09Context],
    pub limits: crate::I09Limits,
}

pub struct DeviceFeaturePlan {
    specs: Vec<FeatureSpec>,
    schema_id: String,
    policy_id: String,
    device_index: usize,
    evaluations: Option<usize>,
    max_bytes: Option<usize>,
    provider_count: usize,
    contexts: Option<PreparedContexts>,
}

fn context_free_supported(id: FeatureId) -> bool {
    matches!(
        (id.family(), id.ordinal()),
        ('A', 1..=20) | ('B', 1..=12) | ('C', 1) | ('D', 3) | ('F', 1) | ('G', 5) | ('H', 1..=6)
    )
}

fn selected(specs: &[FeatureSpec], family: char, ordinals: &[u8]) -> bool {
    specs
        .iter()
        .any(|s| s.id.family() == family && ordinals.contains(&s.id.ordinal()))
}

impl DeviceFeaturePlan {
    pub(crate) fn has_groups(&self) -> bool {
        self.specs
            .iter()
            .any(|s| s.role == crate::FeatureRole::Group)
    }

    /// Compile metadata once. CPU is used only to check the selected static
    /// contracts; no state, feature value, or candidate is evaluated here.
    pub fn context_free(
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        device_index: usize,
    ) -> Result<Self, String> {
        let missing: Vec<String> = policy
            .enabled_ids
            .iter()
            .filter(|id| !context_free_supported(**id))
            .map(ToString::to_string)
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "GPU context-free plan has no producer for: {}",
                missing.join(", ")
            ));
        }
        let mut expected = FeatureRegistry::new();
        crate::register_symmetry_alignment_h06(&mut expected).map_err(|error| error.0)?;
        Self::bind(registry, policy, device_index, expected, None)
    }

    /// Bind the implemented physical probes to their exact source identities.
    /// Contexts are uploaded once at setup; build performs no metadata uploads.
    /// All 102 model inputs have producers. H07 is a post-forward diagnostic.
    /// The complete objective, optimizer and search have separate runtime gates.
    pub fn with_contexts(
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        dimension: u32,
        device_index: usize,
        inputs: DeviceFeatureContextInputs<'_>,
    ) -> Result<Self, String> {
        Self::with_contexts_f08_limit(registry, policy, dimension, device_index, inputs, None, None, None, None)
    }

    pub(crate) fn with_contexts_f08_limit(
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        dimension: u32,
        device_index: usize,
        inputs: DeviceFeatureContextInputs<'_>,
        f08_max_n: Option<u32>,
        f08_max_changed_rows: Option<u32>,
        f08_max_component_events: Option<u32>,
        f08_max_history_counts: Option<u32>,
    ) -> Result<Self, String> {
        let missing: Vec<String> = policy
            .enabled_ids
            .iter()
            .filter(|id| {
                !context_free_supported(**id)
                    && !matches!(
                        (id.family(), id.ordinal()),
                        ('C', 2..=10)
                            | ('D', 1..=14)
                            | ('E', 1..=9)
                            | ('F', 2..=10)
                            | ('G', 1..=4 | 6..=10)
                            | ('I', 1..=11)
                    )
            })
            .map(ToString::to_string)
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "GPU feature plan has no producer for: {}",
                missing.join(", ")
            ));
        }
        // Check policy/registry admissibility before uploading static contexts.
        registry.selected_specs(policy).map_err(|error| error.0)?;
        crate::torch_device::require_cuda_device(device_index)?;
        if dimension == 0 || dimension > (1 << 24) {
            return Err(
                "GPU contextual plan requires a positive width within exact f32 rank range".into(),
            );
        }
        let mut expected = FeatureRegistry::new();
        crate::register_symmetry_alignment_h06(&mut expected).map_err(|error| error.0)?;
        let _device_guard = crate::cuda::DeviceGuard::enter(device_index)?;
        let contexts = PreparedContexts::new(
            &mut expected,
            policy,
            dimension,
            device_index,
            inputs,
            f08_max_n,
            f08_max_changed_rows,
            f08_max_component_events,
            f08_max_history_counts,
        )?;
        Self::bind(registry, policy, device_index, expected, Some(contexts))
    }

    fn bind(
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        device_index: usize,
        expected: FeatureRegistry,
        contexts: Option<PreparedContexts>,
    ) -> Result<Self, String> {
        let mut specs = registry.selected_specs(policy).map_err(|error| error.0)?;
        specs.sort_by_key(|s| s.id.index());
        let expected = expected.specs();
        for spec in &specs {
            if *spec != expected[spec.id.index()] {
                return Err(format!(
                    "{} producer metadata differs from the implemented CUDA contract",
                    spec.id
                ));
            }
        }
        crate::torch_device::require_cuda_device(device_index)?;
        Ok(Self {
            evaluations: policy.evaluations,
            max_bytes: policy.max_bytes,
            provider_count: specs
                .iter()
                .filter(|s| !crate::structural::is_builtin(s.id))
                .count(),
            specs,
            schema_id: policy.schema_id.clone(),
            policy_id: policy.id().map_err(|error| error.0)?,
            device_index,
            contexts,
        })
    }

    /// All state-dependent work stays on CUDA. Tensor handles own the queued
    /// producer outputs through model consumption; the host assembles metadata.
    pub fn build(&self, states: &DeviceStateBatch) -> Result<DeviceFeaturePacket, String> {
        self.build_with_root_f08(states, None)
    }

    /// Compute one selected feature without materializing the packet matrix.
    /// This uses the same width applicability and provider status path as build().
    pub(crate) fn build_feature_entry(
        &self,
        states: &DeviceStateBatch,
        id: FeatureId,
    ) -> Result<DeviceFeatureEntry, String> {
        let _phase = crate::stage_profile::stage("single_feature_packet");
        let spec = self
            .specs
            .iter()
            .find(|spec| spec.id == id)
            .ok_or_else(|| format!("{id} is not selected by the GPU feature plan"))?;
        let device = Device::Cuda(self.device_index);
        if states.tensor().device() != device {
            return Err("GPU feature plan and state batch use different devices".into());
        }
        if self
            .contexts
            .as_ref()
            .is_some_and(|contexts| contexts.dimension != states.dimension())
        {
            return Err("GPU contextual plan and state batch use different widths".into());
        }
        if self
            .contexts
            .as_ref()
            .is_some_and(|contexts| contexts.unsupported_at_width.contains(&id))
        {
            return Ok(budget::unavailable(
                spec,
                states.batch_size(),
                i64::from(states.dimension()),
                device,
                DeviceFeatureStatus::Unsupported,
            ));
        }
        let _device_guard = crate::cuda::DeviceGuard::enter(self.device_index)?;
        let (_, mut entries) =
            self.compute_available(states, std::slice::from_ref(spec), None, false)?;
        if entries.len() != 1 || entries[0].spec() != spec {
            return Err(format!("{id} single-feature computation returned a different contract"));
        }
        Ok(entries.remove(0))
    }

    pub(crate) fn build_with_root_f08(
        &self,
        states: &DeviceStateBatch,
        cached_f08: Option<&DeviceFeatureEntry>,
    ) -> Result<DeviceFeaturePacket, String> {
        let _phase = crate::stage_profile::stage("feature_packet");
        let device = Device::Cuda(self.device_index);
        if states.tensor().device() != device {
            return Err("GPU feature plan and state batch use different devices".into());
        }
        if self
            .contexts
            .as_ref()
            .is_some_and(|contexts| contexts.dimension != states.dimension())
        {
            return Err("GPU contextual plan and state batch use different widths".into());
        }
        let _device_guard = crate::cuda::DeviceGuard::enter(self.device_index)?;
        let (matrix, entries) = self.scheduled(states, cached_f08)?;
        DeviceFeaturePacket::new(
            matrix,
            Tensor::ones(
                [states.batch_size(), i64::from(states.dimension())],
                (Kind::Bool, device),
            ),
            self.schema_id.clone(),
            self.policy_id.clone(),
            entries,
        )
        .map_err(|e| e.0)
    }

    fn compute(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        cached_f08: Option<&DeviceFeatureEntry>,
    ) -> Result<(Tensor, Vec<DeviceFeatureEntry>), String> {
        if let Some(contexts) = self
            .contexts
            .as_ref()
            .filter(|c| !c.unsupported_at_width.is_empty())
        {
            let (missing, active): (Vec<_>, Vec<_>) = specs
                .iter()
                .cloned()
                .partition(|s| contexts.unsupported_at_width.contains(&s.id));
            let (matrix, mut entries) =
                self.compute_available(states, &active, cached_f08, true)?;
            let device = states.tensor().device();
            entries.extend(missing.iter().map(|s| {
                budget::unavailable(
                    s,
                    states.batch_size(),
                    i64::from(states.dimension()),
                    device,
                    DeviceFeatureStatus::Unsupported,
                )
            }));
            Ok((
                matrix.ok_or("feature packet matrix was not materialized")?,
                entries,
            ))
        } else {
            let (matrix, entries) = self.compute_available(states, specs, cached_f08, true)?;
            Ok((
                matrix.ok_or("feature packet matrix was not materialized")?,
                entries,
            ))
        }
    }

    fn compute_available(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        cached_f08: Option<&DeviceFeatureEntry>,
        include_matrix: bool,
    ) -> Result<(Option<Tensor>, Vec<DeviceFeatureEntry>), String> {
        let device = states.tensor().device();
        let status = Tensor::full(
            [states.batch_size()],
            DeviceFeatureStatus::Computed as i64,
            (Kind::Uint8, device),
        );
        let mut entries = Vec::with_capacity(specs.len());
        let needs_a = specs
            .iter()
            .any(|s| context_free_supported(s.id) || (s.id.family() == 'G' && s.id.ordinal() <= 6));
        let matrix = if needs_a {
            let a = states.family_a()?;
            let b = if specs.iter().any(|s| s.id.family() == 'B') {
                Some(states.family_b_from_a(a.shallow_clone())?)
            } else {
                None
            };
            let slice = if specs
                .iter()
                .any(|s| context_free_supported(s.id) && !matches!(s.id.family(), 'A' | 'B'))
            {
                Some(states.feature_slice_from_a(a.shallow_clone(), selected(specs, 'H', &[6]))?)
            } else {
                None
            };
            for spec in specs.iter().filter(|spec| context_free_supported(spec.id)) {
                let values = match spec.id.family() {
                    'A' => a.feature(spec.id.ordinal() as usize),
                    'B' => b
                        .as_ref()
                        .and_then(|family| family.feature(spec.id.ordinal() as usize)),
                    _ => slice
                        .as_ref()
                        .and_then(|features| features.feature(spec.id)),
                }
                .ok_or_else(|| format!("{} admitted CUDA producer returned no tensor", spec.id))?;
                entries.push(DeviceFeatureEntry::new(
                    spec.clone(),
                    values.shallow_clone(),
                    status.shallow_clone(),
                    None,
                    None,
                ));
            }
            if let Some(contexts) = &self.contexts {
                if let Some(diffusion) = contexts
                    .column_diffusion
                    .as_ref()
                    .filter(|_| selected(specs, 'G', &[1, 2, 3, 4]))
                {
                    diffusion.append_entries(states, &a, specs, &mut entries)?;
                }
                if let Some(spec) = specs
                    .iter()
                    .find(|s| (s.id.family(), s.id.ordinal()) == ('G', 6))
                {
                    let diffusion = contexts.pair_diffusion.as_ref().expect("bound G06");
                    entries.push(diffusion.project(states, &a, spec)?);
                }
            }
            Some(
                a.feature(1)
                    .ok_or("A01 matrix tensor is missing")?
                    .squeeze_dim(-1),
            )
        } else if include_matrix {
            Some(
                crate::device_state::unpack_matrix(
                    states.tensor(),
                    i64::from(states.dimension()),
                )
                .to_kind(Kind::Float),
            )
        } else {
            None
        };
        if let Some(contexts) = &self.contexts {
            contexts.append_entries(states, specs, &status, &mut entries, cached_f08)?;
            if let Some(shared_word) = contexts
                .shared_word
                .as_ref()
                .filter(|_| selected(specs, 'G', &[7, 8, 9, 10]))
            {
                shared_word.append_entries(states, specs, &mut entries)?;
            }
        }
        Ok((matrix, entries))
    }
}
