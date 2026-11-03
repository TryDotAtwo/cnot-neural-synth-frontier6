//! Exact finite-template response features from the research-v3 response contract.
//!
//! E02 is the expected physical cost at the all-half Bernoulli point for an
//! explicit, ordered source-template choice set. It is not a distance label or
//! an optimum over the full CNOT alphabet.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, feature_spec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

mod compute;
mod rational;
pub use rational::RationalProbability;

const DEFAULT_MAX_WIDTH: usize = 8;
const DEFAULT_MAX_ASSIGNMENTS: usize = 256;
const DEFAULT_MAX_EVALUATIONS: usize = 4096;
const DEFAULT_MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemplateChoiceKind {
    Add,
    Swap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemplateChoice {
    pub kind: TemplateChoiceKind,
    pub control: usize,
    pub target: usize,
}

impl TemplateChoice {
    pub fn new(kind: TemplateChoiceKind, control: usize, target: usize) -> Self {
        Self {
            kind,
            control,
            target,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemplateResponseBudgets {
    pub max_width: usize,
    pub max_assignments: usize,
    pub max_evaluations: usize,
    pub max_bytes: usize,
}

impl Default for TemplateResponseBudgets {
    fn default() -> Self {
        Self {
            max_width: DEFAULT_MAX_WIDTH,
            max_assignments: DEFAULT_MAX_ASSIGNMENTS,
            max_evaluations: DEFAULT_MAX_EVALUATIONS,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplateResponseConfig {
    choices: Vec<TemplateChoice>,
    points: Vec<Vec<RationalProbability>>,
    budgets: TemplateResponseBudgets,
}

impl TemplateResponseConfig {
    pub fn new(
        choices: Vec<TemplateChoice>,
        points: Vec<Vec<RationalProbability>>,
    ) -> Result<Self, FeatureBuildError> {
        Self::with_budgets(choices, points, TemplateResponseBudgets::default())
    }

    pub fn with_budgets(
        choices: Vec<TemplateChoice>,
        points: Vec<Vec<RationalProbability>>,
        budgets: TemplateResponseBudgets,
    ) -> Result<Self, FeatureBuildError> {
        if !(2..=8).contains(&choices.len())
            || choices.iter().any(|choice| choice.control == choice.target)
        {
            return Err(FeatureBuildError(
                "template response requires two to eight ordered choices with distinct wires"
                    .into(),
            ));
        }
        if points.is_empty() || points.iter().any(|point| point.len() != choices.len()) {
            return Err(FeatureBuildError(
                "template response requires nonempty rational probe points matching the choices"
                    .into(),
            ));
        }
        if budgets.max_width == 0
            || budgets.max_assignments == 0
            || budgets.max_evaluations == 0
            || budgets.max_bytes == 0
        {
            return Err(FeatureBuildError(
                "template response budgets must be positive".into(),
            ));
        }
        Ok(Self {
            choices,
            points,
            budgets,
        })
    }
}

const TEMPLATE_RESPONSE_CACHE_CAPACITY: usize = 32;
const TEMPLATE_RESPONSE_IDS: [u8; 6] = [2, 3, 6, 7, 8, 9];
const TEMPLATE_RESPONSE_WIRE_ROLES: [&str; 4] = [
    "first_control",
    "first_target",
    "second_control",
    "second_target",
];

struct TemplateResponseProducer {
    id: FeatureId,
    spec: FeatureSpec,
    shared: Arc<TemplateResponseCache>,
}

struct TemplateResponseCache {
    config: TemplateResponseConfig,
    entries: Mutex<VecDeque<(StateIdentity, CachedTemplateResponse)>>,
}

#[derive(Clone)]
enum CachedTemplateResponse {
    Computed(TemplateResponseValues),
    Unavailable(FeatureStatus, String),
}

#[derive(Clone)]
struct TemplateResponseValues {
    e02: FeatureValue,
    e03: FeatureValue,
    e06: FeatureValue,
    e07: FeatureValue,
    e08: FeatureValue,
    e09: FeatureValue,
}

impl TemplateResponseValues {
    fn for_id(&self, id: FeatureId) -> FeatureValue {
        match id.ordinal() {
            2 => self.e02.clone(),
            3 => self.e03.clone(),
            6 => self.e06.clone(),
            7 => self.e07.clone(),
            8 => self.e08.clone(),
            9 => self.e09.clone(),
            _ => unreachable!("only template response IDs share this provider"),
        }
    }
}

impl TemplateResponseCache {
    fn get(&self, state: &MatrixState) -> CachedTemplateResponse {
        let key = StateIdentity::from(state);
        let Ok(mut entries) = self.entries.lock() else {
            return CachedTemplateResponse::Unavailable(
                FeatureStatus::ResourceLimit,
                "template_response_cache_poisoned".into(),
            );
        };
        if let Some(index) = entries.iter().position(|(cached, _)| cached == &key) {
            let entry = entries.remove(index).expect("located cache entry");
            let result = entry.1.clone();
            entries.push_back(entry);
            return result;
        }

        let result = compute_template_response(state, &self.config);
        if entries.len() == TEMPLATE_RESPONSE_CACHE_CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, result.clone()));
        result
    }
}

impl FeatureProducer for TemplateResponseProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != self.symmetry_contract() {
            return unavailable(
                FeatureStatus::Unsupported,
                "template_response_requires_augmentation_only_symmetry",
            );
        }
        if policy.max_bytes.is_some_and(|limit| {
            template_response_workspace_bytes(state.dimension(), self.shared.config.choices.len())
                .map_or(true, |bytes| bytes > limit)
        }) {
            return resource_limit("template_response_workspace_budget_exhausted");
        }
        match self.shared.get(state) {
            CachedTemplateResponse::Computed(values) => {
                FeatureComputation::Computed(values.for_id(self.id))
            }
            CachedTemplateResponse::Unavailable(status, reason) => {
                FeatureComputation::Unavailable { status, reason }
            }
        }
    }
}

use compute::{
    compute_template_response, resource_limit, template_response_workspace_bytes, unavailable,
};

/// Registers the source-defined E02/E03/E06-E09 finite-template feature family.
/// All six providers share one bounded per-state response cache.
pub fn register_template_response(
    registry: &mut FeatureRegistry,
    config: TemplateResponseConfig,
) -> Result<(), FeatureBuildError> {
    for ordinal in TEMPLATE_RESPONSE_IDS {
        let id = FeatureId::new(b'E', ordinal).expect("template response ID is in the manifest");
        if registry.specs()[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let shared = Arc::new(TemplateResponseCache {
        config: config.clone(),
        entries: Mutex::new(VecDeque::new()),
    });
    let digest = config_sha256(&config);
    for ordinal in TEMPLATE_RESPONSE_IDS {
        let id = FeatureId::new(b'E', ordinal).expect("template response ID is in the manifest");
        let mut spec = feature_spec(id);
        spec.role = if ordinal == 6 {
            FeatureRole::Group
        } else {
            FeatureRole::Global
        };
        spec.channels = match ordinal {
            3 => config.points.len(),
            8 => 2,
            9 => 5,
            _ => 1,
        };
        spec.scope =
            "selected_partial_shear;fixed_source_template;exact_finite_assignment_scope_only"
                .into();
        spec.producer_version = format!("finite_exact_template_response_e{ordinal:02}_v2:{digest}");
        spec.dependencies = vec![
            "M".into(),
            "deterministic_partial_shear".into(),
            "fixed_source_template_choices".into(),
            "rational_probe_points".into(),
            "exact_response_budgets".into(),
        ];
        spec.exactness = "float32_rounded_from_exact_rational".into();
        spec.cost_tier = "T2".into();
        spec.wire_roles = if ordinal == 6 {
            TEMPLATE_RESPONSE_WIRE_ROLES
                .iter()
                .map(|role| (*role).into())
                .collect()
        } else {
            Vec::new()
        };
        registry.register(Box::new(TemplateResponseProducer {
            id,
            spec,
            shared: Arc::clone(&shared),
        }))?;
    }
    Ok(())
}

/// Backward-compatible entry point; registration now includes the complete
/// template-response family rather than E02 alone.
pub fn register_template_response_e02(
    registry: &mut FeatureRegistry,
    config: TemplateResponseConfig,
) -> Result<(), FeatureBuildError> {
    register_template_response(registry, config)
}

fn config_sha256(config: &TemplateResponseConfig) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"finite_exact_template_response_family_v2");
    hash_field(&mut digest, &(config.choices.len() as u64).to_le_bytes());
    for choice in &config.choices {
        hash_field(
            &mut digest,
            match choice.kind {
                TemplateChoiceKind::Add => b"add",
                TemplateChoiceKind::Swap => b"swap",
            },
        );
        hash_field(&mut digest, &(choice.control as u64).to_le_bytes());
        hash_field(&mut digest, &(choice.target as u64).to_le_bytes());
    }
    hash_field(&mut digest, &(config.points.len() as u64).to_le_bytes());
    for point in &config.points {
        hash_field(&mut digest, &(point.len() as u64).to_le_bytes());
        for probability in point {
            hash_field(&mut digest, &probability.numerator.to_le_bytes());
            hash_field(&mut digest, &probability.denominator.to_le_bytes());
        }
    }
    hash_field(
        &mut digest,
        &(config.budgets.max_width as u64).to_le_bytes(),
    );
    hash_field(
        &mut digest,
        &(config.budgets.max_assignments as u64).to_le_bytes(),
    );
    hash_field(
        &mut digest,
        &(config.budgets.max_evaluations as u64).to_le_bytes(),
    );
    hash_field(
        &mut digest,
        &(config.budgets.max_bytes as u64).to_le_bytes(),
    );
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}
