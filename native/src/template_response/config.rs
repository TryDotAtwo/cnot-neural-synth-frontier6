//! Public ordered-template inputs and resource limits.

use super::rational::RationalProbability;
use crate::features::FeatureBuildError;

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
    pub(super) choices: Vec<TemplateChoice>,
    pub(super) points: Vec<Vec<RationalProbability>>,
    pub(super) budgets: TemplateResponseBudgets,
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
