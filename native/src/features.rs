//! Fail-closed feature registry and padded packet assembly.
//!
//! Provider code is deliberately absent until an exact Rust implementation is
//! ported and reviewed. Missing providers are recorded as unavailable; they are
//! never replaced by zero-valued features marked as computed.

use crate::packet::{FeatureEntry, FeaturePacket, FeaturePacketError, FeatureStatus};
use crate::schema::{
    all_feature_specs, all_model_input_specs, feature_spec, is_reserved_native_layout, FeatureId,
    FeatureRole, FeatureSpec, FEATURE_COUNT, FEATURE_ID_MANIFEST_ID, FEATURE_LAYOUT_ID,
};
use crate::state::MatrixState;
use crate::structural;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureBuildError(pub String);

impl fmt::Display for FeatureBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FeatureBuildError {}

impl From<FeaturePacketError> for FeatureBuildError {
    fn from(error: FeaturePacketError) -> Self {
        Self(error.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePolicy {
    pub enabled_ids: Vec<FeatureId>,
    pub seed: u64,
    pub schema_id: String,
    pub solver_assisted: bool,
    pub symmetry_contract: String,
    pub evaluations: Option<usize>,
    pub max_bytes: Option<usize>,
}

impl FeaturePolicy {
    pub fn complete_103(schema_id: impl Into<String>) -> Self {
        Self {
            enabled_ids: all_feature_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect(),
            seed: 0,
            schema_id: schema_id.into(),
            solver_assisted: false,
            symmetry_contract: "exact_relabel".into(),
            evaluations: None,
            max_bytes: None,
        }
    }

    /// Selects the canonical 103-ID manifest with the weaker symmetry
    /// contract needed by seeded Monte Carlo and physical-wire probes.
    pub fn complete_103_augmentation_only(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::complete_103(schema_id);
        policy.symmetry_contract = "augmentation_only".into();
        policy
    }

    /// Select the executable model inputs from the canonical 103-ID manifest.
    /// H07 remains a post-forward diagnostic, not a model input.
    pub fn all_model_inputs(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::complete_103(schema_id);
        policy.enabled_ids = all_model_input_specs()
            .into_iter()
            .map(|spec| spec.id)
            .collect();
        policy
    }

    /// All model inputs with the weaker symmetry contract needed by seeded
    /// Monte Carlo and physical-wire probes.
    pub fn all_model_inputs_augmentation_only(schema_id: impl Into<String>) -> Self {
        let mut policy = Self::all_model_inputs(schema_id);
        policy.symmetry_contract = "augmentation_only".into();
        policy
    }

    pub fn id(&self) -> Result<String, FeatureBuildError> {
        validate_policy(self)?;
        // Stable, length-delimited textual form suitable for logs and checkpoint binding.
        // The campaign layer adds a cryptographic digest when it commits artifacts.
        let mut body = String::new();
        push_field(&mut body, &self.schema_id);
        push_field(&mut body, &self.seed.to_string());
        push_field(
            &mut body,
            if self.solver_assisted {
                "solver=1"
            } else {
                "solver=0"
            },
        );
        push_field(&mut body, &self.symmetry_contract);
        push_field(
            &mut body,
            &self
                .evaluations
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        push_field(
            &mut body,
            &self
                .max_bytes
                .map_or_else(|| "none".into(), |v| v.to_string()),
        );
        for id in &self.enabled_ids {
            push_field(&mut body, &id.to_string());
        }
        Ok(body)
    }
}

fn push_field(output: &mut String, value: &str) {
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}

fn validate_policy(policy: &FeaturePolicy) -> Result<(), FeatureBuildError> {
    if policy.enabled_ids.is_empty() {
        return Err(FeatureBuildError(
            "feature policy selects no feature IDs".into(),
        ));
    }
    if policy.schema_id.trim().is_empty() || policy.symmetry_contract.trim().is_empty() {
        return Err(FeatureBuildError(
            "feature policy needs schema and symmetry identities".into(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(policy.enabled_ids.len());
    if policy.enabled_ids.iter().any(|id| !seen.insert(*id)) {
        return Err(FeatureBuildError(
            "feature policy repeats a feature ID".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct FeatureValue {
    /// Shape excluding the batch axis.
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    /// Group-only metadata. `incidence` is `[groups, nodes, wire_roles]`.
    pub token_valid: Vec<u8>,
    pub incidence: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum FeatureComputation {
    Computed(FeatureValue),
    /// A computed tensor plus non-numeric per-row provenance that is retained
    /// in the packet sidecar and never appended to model inputs.
    ComputedWithProvenance {
        value: FeatureValue,
        provenance: Vec<String>,
    },
    Unavailable {
        status: FeatureStatus,
        reason: String,
    },
}

pub trait FeatureProducer: Send + Sync {
    fn spec(&self) -> &FeatureSpec;

    /// Names the actual symmetry contract implemented by the provider.
    fn symmetry_contract(&self) -> &str;

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation;
}

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

pub struct FeatureRegistry {
    specs: HashMap<FeatureId, FeatureSpec>,
    providers: HashMap<FeatureId, Box<dyn FeatureProducer>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut specs = all_feature_specs();
        for spec in &mut specs {
            if let Some(metadata) = structural::metadata(spec.id) {
                spec.scope = metadata.scope.into();
                spec.producer_version = metadata.producer_version.into();
                spec.dependencies = metadata
                    .dependencies
                    .iter()
                    .map(|value| (*value).into())
                    .collect();
                spec.exactness = metadata.exactness.into();
                spec.cost_tier = metadata.cost_tier.into();
                spec.wire_roles.clear();
            }
        }
        Self {
            specs: specs.into_iter().map(|spec| (spec.id, spec)).collect(),
            providers: HashMap::new(),
        }
    }

    /// Register all 102 currently defined model-input features.
    ///
    /// The context bundle is explicit because these providers depend on physical
    /// cuts, bases, templates, alphabets, or experiment horizons. H07 is
    /// intentionally not registered: it is currently a post-forward diagnostic
    /// and has no model-input producer contract. Consequently, complete-103
    /// selection still fails closed on H07 alone.
    pub fn with_all_model_input_providers(
        contexts: FeatureProviderContexts,
    ) -> Result<Self, FeatureBuildError> {
        let FeatureProviderContexts {
            max_n,
            labeled_cuts,
            restricted_intersections,
            rank_increment_probes,
            alphabet,
            selected_residual_gates,
            block_parity_contexts,
            max_parity_patterns,
            max_parity_pairs,
            matrix_boundaries,
            max_boundary_ports,
            max_boundary_equations,
            paid_basis_contexts,
            paid_shear_config,
            template_response,
            bilinear_contexts,
            shared_relation_contexts,
            frequency_contexts,
            positive_packet_contexts,
            i09_contexts,
            i09_limits,
            i10_config,
        } = contexts;

        let mut registry = Self::with_default_context_free_providers(max_n)?;
        crate::rank_features::register_labeled_cuts(&mut registry, labeled_cuts)?;
        crate::rank_features::register_restricted_intersections(
            &mut registry,
            restricted_intersections,
        )?;
        crate::rank_features::register_rank_increment_probes(&mut registry, rank_increment_probes)?;
        crate::rank_features::register_selected_residual_ranks(
            &mut registry,
            &alphabet,
            selected_residual_gates,
        )?;
        crate::block_parities::register_block_parities(
            &mut registry,
            block_parity_contexts,
            max_parity_patterns,
            max_parity_pairs,
        )?;
        crate::matrix_boundary::register_matrix_boundaries(
            &mut registry,
            matrix_boundaries,
            max_boundary_ports,
            max_boundary_equations,
        )
        .map_err(|error| FeatureBuildError(error.to_string()))?;
        crate::paid_basis_features::register_paid_basis_features(
            &mut registry,
            paid_basis_contexts,
            paid_shear_config,
        )?;
        crate::template_response::register_template_response(&mut registry, template_response)?;
        crate::bilinear::register_default_bilinear_contexts(&mut registry, bilinear_contexts)?;
        crate::shared_relation::register_default_shared_relation_contexts(
            &mut registry,
            shared_relation_contexts,
        )?;
        crate::frequency_context::register_default_frequency_contexts(
            &mut registry,
            frequency_contexts,
        )?;
        crate::positive_packet::register_default_positive_packet_contexts(
            &mut registry,
            positive_packet_contexts,
        )?;
        crate::i09_quadratic::register_i09_quadratic_response(
            &mut registry,
            i09_contexts,
            i09_limits,
        )?;
        crate::response::register_tiny_i10(&mut registry, i10_config)?;
        Ok(registry)
    }

    /// Build a registry with default providers that need no experiment-specific
    /// physical contexts. max_n bounds the guarded recognizers and diffusion
    /// profile; their resource budgets and numeric settings are versioned by the
    /// providers. Template-response features and I10 remain opt-in because their
    /// exact experiment set and horizon are part of the feature identity.
    ///
    /// Context-bound providers and H07 are intentionally absent. H07 remains a
    /// post-forward diagnostic helper without a model-input producer contract.
    pub fn with_default_context_free_providers(max_n: u32) -> Result<Self, FeatureBuildError> {
        let mut registry = Self::new();
        crate::rank_features::register_partial_shear_rank(&mut registry)?;
        crate::basis_gradient::register_basis_gradient(&mut registry)?;
        crate::stage26::register_stage26_features_with_budget(&mut registry, max_n, 100_000)?;
        crate::bound_features::register_general_compiler_bounds(&mut registry)?;
        crate::guarded_features::register_guarded_pair_f02_view(&mut registry, max_n, 100_000)?;
        crate::guarded_features::register_guarded_pair(&mut registry, max_n, 100_000)?;
        crate::guarded_features::register_guarded_zero_excess(&mut registry, max_n)?;
        crate::guarded_features::register_guarded_one_excess(&mut registry, max_n, 100_000)?;
        let mut diffusion = crate::diffusion::DiffusionConfig::default();
        diffusion.max_n = max_n;
        crate::diffusion::register_diffusion(&mut registry, diffusion)?;
        crate::symmetry_alignment::register_symmetry_alignment_h06(&mut registry)?;
        Ok(registry)
    }

    pub fn register(
        &mut self,
        producer: Box<dyn FeatureProducer>,
    ) -> Result<(), FeatureBuildError> {
        let spec = producer.spec().clone();
        if !is_reserved_native_layout(spec.id) {
            let canonical = feature_spec(spec.id);
            if spec.role != canonical.role || spec.channels != canonical.channels {
                return Err(FeatureBuildError(format!(
                    "{} provider differs from the canonical feature layout",
                    spec.id
                )));
            }
        }
        if spec.channels == 0 {
            return Err(FeatureBuildError(format!(
                "{} provider must declare at least one channel",
                spec.id
            )));
        }
        if spec.role == FeatureRole::Group {
            let unique_roles: std::collections::HashSet<_> = spec.wire_roles.iter().collect();
            if spec.wire_roles.is_empty()
                || spec.wire_roles.iter().any(|role| role.trim().is_empty())
                || unique_roles.len() != spec.wire_roles.len()
            {
                return Err(FeatureBuildError(format!(
                    "{} group provider must declare unique physical wire roles",
                    spec.id
                )));
            }
        } else if !spec.wire_roles.is_empty() {
            return Err(FeatureBuildError(format!(
                "{} non-group provider cannot declare wire roles",
                spec.id
            )));
        }
        if spec.producer_version.trim().is_empty() || spec.producer_version == "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} provider has no version",
                spec.id
            )));
        }
        if spec.requires_solver_assisted && producer.symmetry_contract().trim().is_empty() {
            return Err(FeatureBuildError(format!(
                "{} solver-assisted provider lacks a contract",
                spec.id
            )));
        }
        if self.providers.contains_key(&spec.id) {
            return Err(FeatureBuildError(format!(
                "{} provider already registered",
                spec.id
            )));
        }
        if self
            .specs
            .get(&spec.id)
            .is_some_and(|existing| existing.producer_version != "unconfigured")
        {
            return Err(FeatureBuildError(format!(
                "{} already has a built-in provider",
                spec.id
            )));
        }
        self.specs.insert(spec.id, spec.clone());
        self.providers.insert(spec.id, producer);
        Ok(())
    }

    /// Returns specs in the canonical 103-ID order, including the actual
    /// producer metadata used by this registry.
    pub fn specs(&self) -> Vec<FeatureSpec> {
        (0..FEATURE_COUNT)
            .filter_map(FeatureId::from_index)
            .map(|id| {
                self.specs
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| feature_spec(id))
            })
            .collect()
    }

    /// Stable digest of every feature layout/provider contract in canonical
    /// manifest order. Provider configuration belongs in `producer_version`.
    pub fn contract_sha256(&self) -> String {
        let mut digest = Sha256::new();
        hash_contract_field(&mut digest, b"cnot103_feature_registry_contract_v1");
        hash_contract_field(&mut digest, FEATURE_ID_MANIFEST_ID.as_bytes());
        hash_contract_field(&mut digest, FEATURE_LAYOUT_ID.as_bytes());
        for spec in self.specs() {
            hash_contract_field(&mut digest, spec.id.to_string().as_bytes());
            hash_contract_field(&mut digest, feature_role_name(spec.role).as_bytes());
            hash_contract_field(&mut digest, &(spec.channels as u64).to_le_bytes());
            hash_contract_field(&mut digest, spec.scope.as_bytes());
            hash_contract_field(&mut digest, spec.producer_version.as_bytes());
            hash_contract_field(&mut digest, &(spec.dependencies.len() as u64).to_le_bytes());
            for dependency in &spec.dependencies {
                hash_contract_field(&mut digest, dependency.as_bytes());
            }
            hash_contract_field(&mut digest, spec.exactness.as_bytes());
            hash_contract_field(&mut digest, spec.cost_tier.as_bytes());
            hash_contract_field(&mut digest, &(spec.wire_roles.len() as u64).to_le_bytes());
            for role in &spec.wire_roles {
                hash_contract_field(&mut digest, role.as_bytes());
            }
            hash_contract_field(&mut digest, &[u8::from(spec.requires_solver_assisted)]);
            let symmetry = self
                .providers
                .get(&spec.id)
                .map(|provider| provider.symmetry_contract())
                .unwrap_or_else(|| {
                    if structural::is_builtin(spec.id) {
                        "exact_relabel"
                    } else {
                        "unavailable"
                    }
                });
            hash_contract_field(&mut digest, symmetry.as_bytes());
        }
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Returns the policy's selected feature contracts in policy order.
    ///
    /// A model must not be constructed for a selected feature that has no
    /// executable native provider. All selected contract failures are returned
    /// together so one missing provider does not hide the rest. Built-in
    /// structural features are computed by the shared structural engine and
    /// therefore count as implemented.
    pub fn selected_specs(
        &self,
        policy: &FeaturePolicy,
    ) -> Result<Vec<FeatureSpec>, FeatureBuildError> {
        validate_policy(policy)?;
        let mut selected = Vec::with_capacity(policy.enabled_ids.len());
        let mut failures = Vec::new();
        for id in &policy.enabled_ids {
            let Some(spec) = self.specs.get(id) else {
                failures.push(format!("{id}: unknown feature ID"));
                continue;
            };

            let mut reasons = Vec::new();
            if spec.requires_solver_assisted && !policy.solver_assisted {
                reasons.push("requires solver-assisted policy");
            }
            if policy.symmetry_contract == "exact_relabel"
                && self
                    .providers
                    .get(id)
                    .is_some_and(|provider| provider.symmetry_contract() != "exact_relabel")
            {
                reasons.push("provider symmetry contract mismatch");
            }
            if spec.producer_version.trim().is_empty()
                || spec.producer_version == "unconfigured"
                || (!self.providers.contains_key(id) && !structural::is_builtin(*id))
            {
                reasons.push("no executable native provider");
            }

            if reasons.is_empty() {
                selected.push(spec.clone());
            } else {
                failures.push(format!("{id}: {}", reasons.join(", ")));
            }
        }
        if failures.is_empty() {
            Ok(selected)
        } else {
            Err(FeatureBuildError(format!(
                "feature policy is not executable: {}",
                failures.join("; ")
            )))
        }
    }

    pub fn build(
        &self,
        states: &[MatrixState],
        policy: &FeaturePolicy,
    ) -> Result<FeaturePacket, FeatureBuildError> {
        validate_policy(policy)?;
        if states.is_empty() {
            return Err(FeatureBuildError(
                "cannot build an empty feature batch".into(),
            ));
        }
        for (index, state) in states.iter().enumerate() {
            if state.alphabet_id().trim().is_empty() {
                return Err(FeatureBuildError(format!(
                    "state {index} has no alphabet identity"
                )));
            }
        }
        for id in &policy.enabled_ids {
            let spec = self
                .specs
                .get(id)
                .ok_or_else(|| FeatureBuildError(format!("unknown feature ID {id}")))?;
            if spec.requires_solver_assisted && !policy.solver_assisted {
                return Err(FeatureBuildError(format!(
                    "{id} requires a solver-assisted feature policy"
                )));
            }
            if policy.symmetry_contract == "exact_relabel" {
                if let Some(provider) = self.providers.get(id) {
                    if provider.symmetry_contract() != "exact_relabel" {
                        return Err(FeatureBuildError(format!(
                            "{id} provider symmetry contract mismatch"
                        )));
                    }
                }
            }
        }

        let nodes = states
            .iter()
            .map(|state| state.dimension() as usize)
            .max()
            .unwrap_or(0);
        let valid_nodes = states
            .iter()
            .flat_map(|state| {
                (0..nodes).map(move |node| u8::from(node < state.dimension() as usize))
            })
            .collect::<Vec<_>>();
        let estimated_bytes = states
            .iter()
            .try_fold(0usize, |total, state| {
                let n = state.dimension() as usize;
                n.checked_mul(n)
                    .and_then(|square| square.checked_mul(1024))
                    .and_then(|matrix| {
                        n.checked_mul(256)
                            .and_then(|vectors| matrix.checked_add(vectors))
                    })
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| FeatureBuildError("feature workspace estimate overflow".into()))?;
        let budget_exhausted_before_compute = policy
            .max_bytes
            .is_some_and(|limit| estimated_bytes > limit);

        // The source implementation computes its structural bundle once per
        // state; selected IDs are read from the shared result.
        let needs_structural = !budget_exhausted_before_compute
            && policy
                .enabled_ids
                .iter()
                .any(|id| structural::is_builtin(*id));
        let structural_rows = if needs_structural {
            states
                .iter()
                .map(|state| {
                    structural::compute(state, &policy.enabled_ids)
                        .map(Some)
                        .map_err(FeatureBuildError)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; states.len()]
        };

        let enabled = policy
            .enabled_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut remaining_evaluations = policy.evaluations;
        let mut entries = Vec::with_capacity(FEATURE_COUNT);
        for index in 0..FEATURE_COUNT {
            let id = FeatureId::from_index(index)
                .ok_or_else(|| FeatureBuildError("feature manifest index invalid".into()))?;
            let spec = self
                .specs
                .get(&id)
                .cloned()
                .unwrap_or_else(|| feature_spec(id));
            entries.push(self.build_entry(
                spec,
                states,
                nodes,
                enabled.contains(&id),
                budget_exhausted_before_compute,
                &mut remaining_evaluations,
                policy,
                &structural_rows,
            )?);
        }
        let policy_id = policy.id()?;
        FeaturePacket::new(
            entries,
            states,
            nodes,
            valid_nodes,
            policy.schema_id.clone(),
            policy_id,
        )
        .map_err(FeatureBuildError::from)
    }

    fn build_entry(
        &self,
        spec: FeatureSpec,
        states: &[MatrixState],
        nodes: usize,
        enabled: bool,
        budget_exhausted: bool,
        remaining_evaluations: &mut Option<usize>,
        policy: &FeaturePolicy,
        structural_rows: &[Option<HashMap<FeatureId, FeatureValue>>],
    ) -> Result<FeatureEntry, FeatureBuildError> {
        let producer = self.providers.get(&spec.id);
        let mut results = Vec::with_capacity(states.len());
        let mut max_groups = 0usize;
        for (state_index, state) in states.iter().enumerate() {
            let result = if !enabled {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::NotRequested,
                    reason: "feature_not_selected_by_policy".into(),
                }
            } else if budget_exhausted {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::ResourceLimit,
                    reason: "workspace_estimate_exceeds_policy_limit".into(),
                }
            } else if let Some(value) = structural_rows
                .get(state_index)
                .and_then(Option::as_ref)
                .and_then(|row| row.get(&spec.id))
            {
                FeatureComputation::Computed(value.clone())
            } else if producer.is_none() {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::Unsupported,
                    reason: "no_native_provider_registered".into(),
                }
            } else if remaining_evaluations.is_some_and(|remaining| remaining == 0) {
                FeatureComputation::Unavailable {
                    status: FeatureStatus::ResourceLimit,
                    reason: "feature_evaluation_budget_exhausted".into(),
                }
            } else {
                if let Some(remaining) = remaining_evaluations {
                    *remaining -= 1;
                }
                producer.expect("checked above").compute(state, policy)
            };
            if let FeatureComputation::Computed(value)
            | FeatureComputation::ComputedWithProvenance { value, .. } = &result
            {
                validate_value(&spec, state.dimension() as usize, value)?;
                if spec.role == FeatureRole::Group {
                    max_groups = max_groups.max(value.shape[0]);
                }
                if let FeatureComputation::ComputedWithProvenance { provenance, .. } = &result {
                    if provenance.iter().any(|item| item.trim().is_empty()) {
                        return Err(FeatureBuildError(format!(
                            "{} returned empty provenance metadata",
                            spec.id
                        )));
                    }
                }
            } else if let FeatureComputation::Unavailable { status, reason } = &result {
                if *status == FeatureStatus::Computed || reason.trim().is_empty() {
                    return Err(FeatureBuildError(format!(
                        "{} returned invalid unavailable status",
                        spec.id
                    )));
                }
            }
            results.push(result);
        }

        let shape = feature_shape(&spec, states.len(), nodes, max_groups);
        let total_values = element_count(&shape)?;
        let mut values = vec![0.0f32; total_values];
        let mut available = Vec::with_capacity(states.len());
        let mut status = Vec::with_capacity(states.len());
        let mut reason = Vec::with_capacity(states.len());
        let mut provenance = vec![Vec::new(); states.len()];
        let mut token_valid = if spec.role == FeatureRole::Group {
            vec![
                0u8;
                states
                    .len()
                    .checked_mul(max_groups)
                    .ok_or_else(|| FeatureBuildError("group mask size overflow".into()))?
            ]
        } else {
            Vec::new()
        };
        let role_count = spec.wire_roles.len();
        let incidence_count = states
            .len()
            .checked_mul(max_groups)
            .and_then(|value| value.checked_mul(nodes))
            .and_then(|value| value.checked_mul(role_count))
            .ok_or_else(|| FeatureBuildError("group incidence size overflow".into()))?;
        let mut incidence = if spec.role == FeatureRole::Group {
            vec![0.0f32; incidence_count]
        } else {
            Vec::new()
        };

        for (batch_index, (state, result)) in states.iter().zip(results).enumerate() {
            match result {
                FeatureComputation::Unavailable {
                    status: row_status,
                    reason: row_reason,
                } => {
                    available.push(0);
                    status.push(row_status);
                    reason.push(row_reason);
                }
                FeatureComputation::Computed(value) => {
                    available.push(1);
                    status.push(FeatureStatus::Computed);
                    reason.push(String::new());
                    copy_feature_row(
                        &spec,
                        &value,
                        batch_index,
                        state.dimension() as usize,
                        nodes,
                        max_groups,
                        &mut values,
                        &mut token_valid,
                        &mut incidence,
                    )?;
                }
                FeatureComputation::ComputedWithProvenance {
                    value,
                    provenance: row_provenance,
                } => {
                    available.push(1);
                    status.push(FeatureStatus::Computed);
                    reason.push(String::new());
                    provenance[batch_index] = row_provenance;
                    copy_feature_row(
                        &spec,
                        &value,
                        batch_index,
                        state.dimension() as usize,
                        nodes,
                        max_groups,
                        &mut values,
                        &mut token_valid,
                        &mut incidence,
                    )?;
                }
            }
        }

        Ok(FeatureEntry {
            spec: spec.clone(),
            shape,
            values,
            available,
            status,
            reason,
            provenance,
            wire_roles: spec.wire_roles.clone(),
            token_valid: (spec.role == FeatureRole::Group).then_some(token_valid),
            incidence: (spec.role == FeatureRole::Group).then_some(incidence),
        })
    }
}

fn feature_role_name(role: FeatureRole) -> &'static str {
    match role {
        FeatureRole::Node => "node",
        FeatureRole::Pair => "pair",
        FeatureRole::Global => "global",
        FeatureRole::Group => "group",
    }
}

fn hash_contract_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn validate_value(
    spec: &FeatureSpec,
    nodes: usize,
    value: &FeatureValue,
) -> Result<(), FeatureBuildError> {
    let expected_rank = match spec.role {
        FeatureRole::Node | FeatureRole::Group => 2,
        FeatureRole::Pair => 3,
        FeatureRole::Global => 1,
    };
    if value.shape.len() != expected_rank || value.shape.last().copied() != Some(spec.channels) {
        return Err(FeatureBuildError(format!(
            "{} provider returned the wrong rank or channel count",
            spec.id
        )));
    }
    let expected_values = element_count(&value.shape)?;
    if value.values.len() != expected_values || value.values.iter().any(|item| !item.is_finite()) {
        return Err(FeatureBuildError(format!(
            "{} provider returned invalid/non-finite values",
            spec.id
        )));
    }
    match spec.role {
        FeatureRole::Node if value.shape[0] != nodes => {
            return Err(FeatureBuildError(format!(
                "{} node extent differs from state",
                spec.id
            )));
        }
        FeatureRole::Pair if value.shape[0] != nodes || value.shape[1] != nodes => {
            return Err(FeatureBuildError(format!(
                "{} pair extent differs from state",
                spec.id
            )));
        }
        FeatureRole::Global => {
            if !value.token_valid.is_empty() || !value.incidence.is_empty() {
                return Err(FeatureBuildError(format!(
                    "{} global feature returned group metadata",
                    spec.id
                )));
            }
        }
        FeatureRole::Group => {
            let groups = value.shape[0];
            let expected_incidence = groups
                .checked_mul(nodes)
                .and_then(|extent| extent.checked_mul(spec.wire_roles.len()))
                .ok_or_else(|| FeatureBuildError("group incidence length overflow".into()))?;
            if value.token_valid.len() != groups
                || value.token_valid.iter().any(|mask| *mask > 1)
                || value.incidence.len() != expected_incidence
                || value
                    .incidence
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(FeatureBuildError(format!(
                    "{} group metadata shape/value mismatch",
                    spec.id
                )));
            }
        }
        _ => {
            if !value.token_valid.is_empty() || !value.incidence.is_empty() {
                return Err(FeatureBuildError(format!(
                    "{} non-group feature returned group metadata",
                    spec.id
                )));
            }
        }
    }
    Ok(())
}

fn feature_shape(spec: &FeatureSpec, batch: usize, nodes: usize, groups: usize) -> Vec<usize> {
    match spec.role {
        FeatureRole::Node => vec![batch, nodes, spec.channels],
        FeatureRole::Pair => vec![batch, nodes, nodes, spec.channels],
        FeatureRole::Global => vec![batch, spec.channels],
        FeatureRole::Group => vec![batch, groups, spec.channels],
    }
}

fn element_count(shape: &[usize]) -> Result<usize, FeatureBuildError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| FeatureBuildError("feature tensor element count overflow".into()))
}

fn copy_feature_row(
    spec: &FeatureSpec,
    value: &FeatureValue,
    batch_index: usize,
    state_nodes: usize,
    max_nodes: usize,
    max_groups: usize,
    values: &mut [f32],
    token_valid: &mut [u8],
    incidence: &mut [f32],
) -> Result<(), FeatureBuildError> {
    match spec.role {
        FeatureRole::Node | FeatureRole::Global => {
            let per_row = if spec.role == FeatureRole::Node {
                max_nodes * spec.channels
            } else {
                spec.channels
            };
            let base = batch_index * per_row;
            values[base..base + value.values.len()].copy_from_slice(&value.values);
        }
        FeatureRole::Pair => {
            let source_dim = state_nodes * state_nodes * spec.channels;
            let source_index = |row: usize, col: usize, channel: usize| {
                (row * state_nodes + col) * spec.channels + channel
            };
            let target_index = |row: usize, col: usize, channel: usize| {
                ((batch_index * max_nodes + row) * max_nodes + col) * spec.channels + channel
            };
            if value.values.len() != source_dim {
                return Err(FeatureBuildError(format!(
                    "{} pair source buffer changed after validation",
                    spec.id
                )));
            }
            for row in 0..state_nodes {
                for col in 0..state_nodes {
                    for channel in 0..spec.channels {
                        values[target_index(row, col, channel)] =
                            value.values[source_index(row, col, channel)];
                    }
                }
            }
        }
        FeatureRole::Group => {
            let groups = value.shape[0];
            let channels = spec.channels;
            let target_base = batch_index * max_groups * channels;
            values[target_base..target_base + value.values.len()].copy_from_slice(&value.values);
            let token_base = batch_index * max_groups;
            token_valid[token_base..token_base + groups].copy_from_slice(&value.token_valid);
            let roles = spec.wire_roles.len();
            for group in 0..groups {
                for node in 0..state_nodes {
                    for role in 0..roles {
                        let source = (group * state_nodes + node) * roles + role;
                        let target =
                            ((batch_index * max_groups + group) * max_nodes + node) * roles + role;
                        incidence[target] = value.incidence[source];
                    }
                }
            }
        }
    }
    Ok(())
}
