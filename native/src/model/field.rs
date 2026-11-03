//! Selected-feature structural distance field and forward contract.

use super::auxiliary::AuxiliaryHeads;
use super::family_heads::FamilyHeads;
use super::fusion::FusionTransformer;
use super::stem::StructuralStem;
use super::{EncodedState, FieldOutput, ModelConfig, ModelError};
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::packet::FeaturePacket;
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID};
use tch::nn;
use tch::Device;

/// The selected-feature structural distance field used by research_v3.
///
/// `specs` is the selected subset of the canonical 103-ID manifest. Built-in
/// shapes are fixed; a reserved ID receives its role and channel count from
/// its registered provider. The packet still carries all 103 manifest rows.
/// The model rejects packets whose selected IDs, producer metadata, or policy
/// differ from the schema selected at construction. Row availability is an
/// input mask: unavailable rows stay zero-masked and do not block other states
/// in a batch. This v24 contract is distinct from the legacy fixed-shape v22
/// packet adapter.
#[derive(Debug)]
pub struct FusionDistanceField {
    specs: Vec<FeatureSpec>,
    expected_schema_id: String,
    expected_policy_id: String,
    device: Device,
    stem: StructuralStem,
    family_heads: FamilyHeads,
    fusion: FusionTransformer,
    auxiliary: AuxiliaryHeads,
}

impl FusionDistanceField {
    /// Constructs a model directly from the registry and feature policy used
    /// to build its packets, so the model contract cannot drift from either.
    pub fn new_for_policy(
        path: &nn::Path<'_>,
        registry: &FeatureRegistry,
        policy: &FeaturePolicy,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        let specs = registry
            .selected_specs(policy)
            .map_err(|error| ModelError(error.0))?;
        let policy_id = policy.id().map_err(|error| ModelError(error.0))?;
        Self::new(
            path,
            &specs,
            policy.schema_id.clone(),
            policy_id,
            device,
            config,
        )
    }

    pub fn new(
        path: &nn::Path<'_>,
        specs: &[FeatureSpec],
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        crate::torch_device::require_device(device).map_err(ModelError)?;
        config.validate()?;
        let schema_id = schema_id.into();
        let policy_id = policy_id.into();
        if schema_id.trim().is_empty() || policy_id.trim().is_empty() {
            return Err(ModelError(
                "model requires explicit schema and policy identities".into(),
            ));
        }
        if specs.is_empty() {
            return Err(ModelError(
                "model requires at least one selected feature spec".into(),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(specs.len());
        let mut families: Vec<_> = specs.iter().map(|spec| spec.id.family()).collect();
        for spec in specs {
            if !seen.insert(spec.id) {
                return Err(ModelError(format!(
                    "{} appears more than once in the model feature schema",
                    spec.id
                )));
            }
            if spec.channels == 0 || i64::try_from(spec.channels).is_err() {
                return Err(ModelError(format!(
                    "{} has an invalid channel count",
                    spec.id
                )));
            }
            if spec.producer_version.trim().is_empty()
                || spec.producer_version == "unconfigured"
                || spec.scope.trim().is_empty()
                || spec.exactness.trim().is_empty()
                || spec.exactness == "unknown"
                || spec.cost_tier.trim().is_empty()
                || spec
                    .dependencies
                    .iter()
                    .any(|value| value.trim().is_empty())
            {
                return Err(ModelError(format!(
                    "{} does not have an accepted producer contract",
                    spec.id
                )));
            }
            if spec.role == FeatureRole::Group
                && (spec.wire_roles.is_empty()
                    || spec.wire_roles.iter().any(|role| role.trim().is_empty())
                    || spec
                        .wire_roles
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != spec.wire_roles.len())
            {
                return Err(ModelError(format!(
                    "{} requires a unique physical wire-role layout",
                    spec.id
                )));
            }
            if spec.role != FeatureRole::Group && !spec.wire_roles.is_empty() {
                return Err(ModelError(format!(
                    "{} has group roles on a non-group feature",
                    spec.id
                )));
            }
        }
        families.sort_unstable();
        families.dedup();

        Ok(Self {
            specs: specs.to_vec(),
            expected_schema_id: schema_id,
            expected_policy_id: policy_id,
            device,
            stem: StructuralStem::new(&(path / "stem"), specs, &config),
            family_heads: FamilyHeads::new(&(path / "family_heads"), &families, &config),
            fusion: FusionTransformer::new(&(path / "fusion"), &config, families.len()),
            auxiliary: AuxiliaryHeads::new(&(path / "auxiliary"), &config),
        })
    }

    pub(crate) fn encode(&self, packet: &FeaturePacket) -> Result<EncodedState, ModelError> {
        if packet.schema_id != self.expected_schema_id
            || packet.policy_id != self.expected_policy_id
        {
            return Err(ModelError(
                "feature packet schema or policy differs from model contract".into(),
            ));
        }
        if packet.feature_id_manifest_id != FEATURE_ID_MANIFEST_ID
            || packet.entries.len() != FEATURE_COUNT
            || packet
                .entries
                .iter()
                .enumerate()
                .any(|(index, entry)| FeatureId::from_index(index) != Some(entry.spec.id))
        {
            return Err(ModelError(format!(
                "feature packet must preserve the canonical {FEATURE_COUNT}-ID manifest"
            )));
        }
        let model_ids: std::collections::HashSet<_> =
            self.specs.iter().map(|spec| spec.id).collect();
        let packet_ids: std::collections::HashSet<_> = packet
            .entries
            .iter()
            .filter(|entry| {
                entry
                    .status
                    .iter()
                    .any(|status| *status != crate::packet::FeatureStatus::NotRequested)
            })
            .map(|entry| entry.spec.id)
            .collect();
        if model_ids != packet_ids {
            return Err(ModelError(
                "selected packet feature IDs differ from model input schema".into(),
            ));
        }
        if self.specs.iter().any(|spec| {
            packet
                .entries
                .get(spec.id.index())
                .is_none_or(|entry| entry.spec != *spec)
        }) {
            return Err(ModelError(
                "feature packet producer schema differs from model contract".into(),
            ));
        }
        self.stem.forward(packet, self.device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub(crate) fn forward_encoded(&self, encoded: &EncodedState) -> FieldOutput {
        let (family_tokens, family_masks) = self.family_heads.forward(&encoded);
        let positive_value = self.fusion.forward(&encoded, &family_tokens, &family_masks);
        let value =
            Tensor::zeros_like(&positive_value).where_self(&encoded.identity, &positive_value);
        let (proposal_logits, qhat, action_mask) =
            self.auxiliary
                .forward(&encoded.wire, &encoded.pair, &value, &encoded.valid);
        FieldOutput {
            value,
            proposal_logits,
            qhat,
            action_mask,
        }
    }

    pub fn forward(&self, packet: &FeaturePacket) -> Result<FieldOutput, ModelError> {
        let encoded = self.encode(packet)?;
        Ok(self.forward_encoded(&encoded))
    }
}
