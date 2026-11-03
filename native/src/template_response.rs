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
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::shear::partial_shear;
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

mod rational;
pub use rational::RationalProbability;
use rational::{rational_to_f32, signed_rational_to_f32};

const DEFAULT_MAX_WIDTH: usize = 8;
const DEFAULT_MAX_ASSIGNMENTS: usize = 256;
const DEFAULT_MAX_EVALUATIONS: usize = 4096;
const DEFAULT_MAX_BYTES: usize = 16 * 1024 * 1024;
const RATIONAL_ITEM_BYTES: usize = 128;

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

fn source_evaluation_estimate(
    choice_count: usize,
    point_count: usize,
    assignment_count: usize,
) -> Option<usize> {
    let pair_count = choice_count.checked_mul(choice_count.saturating_sub(1))? / 2;
    let pair_evaluations = pair_count.checked_mul(4)?;
    let point_rounding_evaluations = point_count
        .checked_add(2)?
        .checked_mul(choice_count.checked_mul(2)?.checked_add(2)?)?;
    1usize
        .checked_add(point_count)?
        .checked_add(pair_evaluations)?
        .checked_add(point_rounding_evaluations)?
        .checked_add(assignment_count)
}

fn compute_template_response(
    state: &MatrixState,
    config: &TemplateResponseConfig,
) -> CachedTemplateResponse {
    let width = (state.dimension() / 2) as usize;
    let choice_count = config.choices.len();
    let Some(assignment_count) = 1usize.checked_shl(choice_count as u32) else {
        return cached_resource_limit("template_response_assignment_count_overflow");
    };
    if width > config.budgets.max_width {
        return cached_resource_limit("template_response_width_budget_exhausted");
    }
    if assignment_count > config.budgets.max_assignments {
        return cached_resource_limit("template_response_assignment_budget_exhausted");
    }
    let Some(workspace_bytes) = template_response_workspace_bytes(state.dimension(), choice_count)
    else {
        return cached_resource_limit("template_response_workspace_estimate_overflow");
    };
    if workspace_bytes > config.budgets.max_bytes {
        return cached_resource_limit("template_response_workspace_budget_exhausted");
    }
    let Some(required_evaluations) =
        source_evaluation_estimate(choice_count, config.points.len(), assignment_count)
    else {
        return cached_resource_limit("template_response_evaluation_estimate_overflow");
    };
    if required_evaluations > config.budgets.max_evaluations {
        return cached_resource_limit("template_response_evaluation_budget_exhausted");
    }

    let decomposition = match partial_shear(state) {
        Ok(Some(value)) => value,
        Ok(None) => {
            return cached_inapplicable("no_nonzero_partial_shear_block");
        }
        Err(_) => return cached_unsupported("partial_shear_decomposition_failed"),
    };
    if config.choices.iter().any(|choice| {
        choice.control >= decomposition.source_wires.len()
            || choice.target >= decomposition.source_wires.len()
    }) {
        return cached_unsupported("template_choice_outside_partial_shear_source_block");
    }

    let mut assignment_costs = Vec::with_capacity(assignment_count);
    for assignment in 0..assignment_count {
        let Some(cost) =
            template_assignment_cost(&decomposition.block, &config.choices, assignment)
        else {
            return cached_resource_limit("template_response_cost_overflow");
        };
        assignment_costs.push(cost);
    }

    let midpoint = vec![
        RationalProbability {
            numerator: 1,
            denominator: 2
        };
        choice_count
    ];
    let Some((e02_numerator, e02_denominator)) =
        expected_cost_fraction(&assignment_costs, &midpoint)
    else {
        return cached_resource_limit("template_response_exact_rational_overflow");
    };
    let Some(e02_value) = rational_to_f32(e02_numerator, e02_denominator) else {
        return cached_resource_limit("template_response_float32_conversion_failed");
    };

    let mut e03_values = Vec::with_capacity(config.points.len());
    for point in &config.points {
        let Some((numerator, denominator)) = expected_cost_fraction(&assignment_costs, point)
        else {
            return cached_resource_limit("template_response_exact_rational_overflow");
        };
        let Some(value) = rational_to_f32(numerator, denominator) else {
            return cached_resource_limit("template_response_float32_conversion_failed");
        };
        e03_values.push(value);
    }

    let mut e06_values = Vec::new();
    let mut e06_incidence = Vec::new();
    for first in 0..choice_count {
        for second in first + 1..choice_count {
            let Some((numerator, denominator)) =
                mixed_difference_at_midpoint(&assignment_costs, first, second)
            else {
                return cached_resource_limit("template_response_mixed_difference_overflow");
            };
            let Some(value) = signed_rational_to_f32(numerator, denominator) else {
                return cached_resource_limit("template_response_float32_conversion_failed");
            };
            e06_values.push(value);

            let wires = [
                decomposition.source_wires[config.choices[first].control],
                decomposition.source_wires[config.choices[first].target],
                decomposition.source_wires[config.choices[second].control],
                decomposition.source_wires[config.choices[second].target],
            ];
            for node in 0..state.dimension() as usize {
                for role in 0..TEMPLATE_RESPONSE_WIRE_ROLES.len() {
                    e06_incidence.push(if wires[role] == node { 1.0 } else { 0.0 });
                }
            }
        }
    }

    let Some(e07_cost) = conditional_round_cost(&assignment_costs, &midpoint) else {
        return cached_resource_limit("template_response_exact_rational_overflow");
    };
    let mut reverse_point = midpoint.clone();
    for index in (0..choice_count).rev() {
        let Some((zero_cost, one_cost)) =
            conditional_choice_totals(&assignment_costs, &reverse_point, index)
        else {
            return cached_resource_limit("template_response_exact_rational_overflow");
        };
        reverse_point[index] = deterministic_probability(one_cost < zero_cost);
    }
    let reverse_bits = reverse_point
        .iter()
        .enumerate()
        .fold(0usize, |mask, (index, probability)| {
            mask | (usize::from(probability.numerator == 1) << index)
        });
    let reverse_cost = assignment_costs[reverse_bits];

    let best_cost = *assignment_costs.iter().min().expect("nonempty assignments");
    let mut multistart_costs = Vec::with_capacity(config.points.len() + 1);
    multistart_costs.push(e07_cost);
    for point in &config.points {
        let Some(cost) = conditional_round_cost(&assignment_costs, point) else {
            return cached_resource_limit("template_response_exact_rational_overflow");
        };
        multistart_costs.push(cost);
    }
    let min_multistart = *multistart_costs
        .iter()
        .min()
        .expect("midpoint start exists");
    let max_multistart = *multistart_costs
        .iter()
        .max()
        .expect("midpoint start exists");

    CachedTemplateResponse::Computed(TemplateResponseValues {
        e02: scalar_feature(e02_value),
        e03: FeatureValue {
            shape: vec![e03_values.len()],
            values: e03_values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
        e06: FeatureValue {
            shape: vec![e06_values.len(), 1],
            values: e06_values,
            token_valid: vec![1; choice_count * (choice_count - 1) / 2],
            incidence: e06_incidence,
        },
        e07: scalar_feature(e07_cost as f32),
        e08: FeatureValue {
            shape: vec![2],
            values: vec![best_cost as f32, 1.0],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
        e09: FeatureValue {
            shape: vec![5],
            values: vec![
                e07_cost as f32,
                reverse_cost as f32,
                e07_cost.abs_diff(reverse_cost) as f32,
                min_multistart as f32,
                max_multistart as f32,
            ],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
    })
}

fn template_assignment_cost(
    block: &[Vec<u8>],
    choices: &[TemplateChoice],
    assignment: usize,
) -> Option<u64> {
    let mut transformed = block.to_vec();
    let mut preparation_cost = 0u64;
    for (index, choice) in choices.iter().copied().enumerate() {
        if assignment & (1usize << index) == 0 {
            continue;
        }
        match choice.kind {
            TemplateChoiceKind::Add => {
                preparation_cost = preparation_cost.checked_add(2)?;
                for row in &mut transformed {
                    row[choice.control] ^= row[choice.target];
                }
            }
            TemplateChoiceKind::Swap => {
                preparation_cost = preparation_cost.checked_add(6)?;
                for row in &mut transformed {
                    row.swap(choice.control, choice.target);
                }
            }
        }
    }
    let residual_cost = transformed
        .iter()
        .flatten()
        .try_fold(0u64, |sum, bit| sum.checked_add(u64::from(*bit)))?;
    preparation_cost.checked_add(residual_cost)
}

fn template_response_workspace_bytes(dimension: u32, choice_count: usize) -> Option<usize> {
    let width = (dimension / 2) as usize;
    1usize
        .checked_shl(width as u32)?
        .checked_mul(choice_count.saturating_add(6))?
        .checked_mul(RATIONAL_ITEM_BYTES)
}

fn scalar_feature(value: f32) -> FeatureValue {
    FeatureValue {
        shape: vec![1],
        values: vec![value],
        token_valid: Vec::new(),
        incidence: Vec::new(),
    }
}

fn expected_cost_fraction(
    costs: &[u64],
    probabilities: &[RationalProbability],
) -> Option<(u128, u128)> {
    if costs.len() != (1usize.checked_shl(probabilities.len() as u32)?) {
        return None;
    }
    let denominator = probabilities
        .iter()
        .try_fold(1u128, |product, probability| {
            product.checked_mul(u128::from(probability.denominator))
        })?;
    let mut numerator = 0u128;
    for (assignment, cost) in costs.iter().copied().enumerate() {
        let mut weight = 1u128;
        for (index, probability) in probabilities.iter().enumerate() {
            let factor = if assignment & (1usize << index) != 0 {
                probability.numerator
            } else {
                probability.denominator - probability.numerator
            };
            weight = weight.checked_mul(u128::from(factor))?;
        }
        numerator = numerator.checked_add(weight.checked_mul(u128::from(cost))?)?;
    }
    Some((numerator, denominator))
}

fn conditional_choice_totals(
    costs: &[u64],
    probabilities: &[RationalProbability],
    fixed_index: usize,
) -> Option<(u128, u128)> {
    if probabilities.len() >= usize::BITS as usize
        || fixed_index >= probabilities.len()
        || costs.len() != (1usize.checked_shl(probabilities.len() as u32)?)
    {
        return None;
    }
    let mut zero_total = 0u128;
    let mut one_total = 0u128;
    for assignment in 0..costs.len() {
        if assignment & (1usize << fixed_index) != 0 {
            continue;
        }
        let mut weight = 1u128;
        for (index, probability) in probabilities.iter().enumerate() {
            if index == fixed_index {
                continue;
            }
            let factor = if assignment & (1usize << index) != 0 {
                probability.numerator
            } else {
                probability.denominator - probability.numerator
            };
            weight = weight.checked_mul(u128::from(factor))?;
        }
        zero_total = zero_total.checked_add(weight.checked_mul(u128::from(costs[assignment]))?)?;
        one_total = one_total.checked_add(
            weight.checked_mul(u128::from(costs[assignment | (1usize << fixed_index)]))?,
        )?;
    }
    Some((zero_total, one_total))
}

fn conditional_round_cost(costs: &[u64], probabilities: &[RationalProbability]) -> Option<u64> {
    let mut point = probabilities.to_vec();
    for index in 0..point.len() {
        let (zero_cost, one_cost) = conditional_choice_totals(costs, &point, index)?;
        point[index] = deterministic_probability(one_cost < zero_cost);
    }
    let assignment = point
        .iter()
        .enumerate()
        .fold(0usize, |mask, (index, probability)| {
            mask | (usize::from(probability.numerator == 1) << index)
        });
    costs.get(assignment).copied()
}

fn mixed_difference_at_midpoint(
    costs: &[u64],
    first: usize,
    second: usize,
) -> Option<(i128, u128)> {
    let bit_count = costs.len().trailing_zeros() as usize;
    if costs.len() != (1usize.checked_shl(bit_count as u32)?)
        || first >= bit_count
        || second >= bit_count
        || first >= second
        || bit_count < 2
    {
        return None;
    }
    let assignment_count = 1usize.checked_shl((bit_count - 2) as u32)?;
    let mut difference_sum = 0i128;
    for reduced in 0..assignment_count {
        let mut base = 0usize;
        let mut reduced_index = 0usize;
        for index in 0..bit_count {
            if index == first || index == second {
                continue;
            }
            base |= ((reduced >> reduced_index) & 1) << index;
            reduced_index += 1;
        }
        let first_bit = 1usize << first;
        let second_bit = 1usize << second;
        let difference = i128::from(costs[base | first_bit | second_bit])
            - i128::from(costs[base | first_bit])
            - i128::from(costs[base | second_bit])
            + i128::from(costs[base]);
        difference_sum = difference_sum.checked_add(difference)?;
    }
    Some((difference_sum, assignment_count as u128))
}

fn deterministic_probability(bit: bool) -> RationalProbability {
    RationalProbability {
        numerator: u64::from(bit),
        denominator: 1,
    }
}

fn cached_unsupported(reason: &str) -> CachedTemplateResponse {
    CachedTemplateResponse::Unavailable(FeatureStatus::Unsupported, reason.into())
}

fn cached_inapplicable(reason: &str) -> CachedTemplateResponse {
    CachedTemplateResponse::Unavailable(FeatureStatus::Inapplicable, reason.into())
}

fn cached_resource_limit(reason: &str) -> CachedTemplateResponse {
    CachedTemplateResponse::Unavailable(FeatureStatus::ResourceLimit, reason.into())
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

fn resource_limit(reason: &str) -> FeatureComputation {
    unavailable(FeatureStatus::ResourceLimit, reason)
}

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
