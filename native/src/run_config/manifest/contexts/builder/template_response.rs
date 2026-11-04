//! Conversion of template-response probabilities, choices, and budgets.

use super::super::{ChoiceKindSpec, TemplateResponseSpec};
use crate::{
    RationalProbability, TemplateChoice, TemplateChoiceKind, TemplateResponseBudgets,
    TemplateResponseConfig,
};

impl TemplateResponseSpec {
    pub(crate) fn build(self) -> Result<TemplateResponseConfig, String> {
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
impl From<ChoiceKindSpec> for TemplateChoiceKind {
    fn from(x: ChoiceKindSpec) -> Self {
        match x {
            ChoiceKindSpec::Add => Self::Add,
            ChoiceKindSpec::Swap => Self::Swap,
        }
    }
}
