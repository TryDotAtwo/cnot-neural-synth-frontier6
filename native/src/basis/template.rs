//! Public template contract shared by basis proposal, compilation, and loss code.

use crate::model::ModelError;
use crate::paid_basis::BasisSide;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisChoiceKind {
    Add,
    Swap,
}

impl BasisChoiceKind {
    pub(super) fn paid_cost(self) -> f32 {
        match self {
            Self::Add => 1.0,
            Self::Swap => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BasisChoice {
    pub kind: BasisChoiceKind,
    pub control: usize,
    pub target: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasisTemplate {
    pub n: usize,
    pub source_wires: Vec<usize>,
    pub target_wires: Vec<usize>,
    pub side: BasisSide,
    pub choices: Vec<BasisChoice>,
}

impl BasisTemplate {
    pub fn new(
        n: usize,
        source_wires: Vec<usize>,
        target_wires: Vec<usize>,
        side: BasisSide,
        choices: Vec<BasisChoice>,
    ) -> Result<Self, ModelError> {
        let template = Self {
            n,
            source_wires,
            target_wires,
            side,
            choices,
        };
        template.validate()?;
        Ok(template)
    }

    pub fn group_wires(&self) -> &[usize] {
        match self.side {
            BasisSide::Source => &self.source_wires,
            BasisSide::Target => &self.target_wires,
        }
    }

    pub fn width(&self) -> usize {
        self.group_wires().len()
    }

    pub(super) fn validate(&self) -> Result<(), ModelError> {
        if self.n == 0 || self.source_wires.is_empty() || self.target_wires.is_empty() {
            return Err(ModelError(
                "basis template requires a positive width and two nonempty wire blocks".into(),
            ));
        }

        let mut physical_wires =
            HashSet::with_capacity(self.source_wires.len() + self.target_wires.len());
        for &wire in self.source_wires.iter().chain(&self.target_wires) {
            if wire >= self.n || !physical_wires.insert(wire) {
                return Err(ModelError(
                    "basis template has an out-of-range or overlapping physical wire".into(),
                ));
            }
        }

        let group_width = self.width();
        for choice in &self.choices {
            if choice.control >= group_width
                || choice.target >= group_width
                || choice.control == choice.target
            {
                return Err(ModelError(
                    "basis choice has invalid local control/target indices".into(),
                ));
            }
        }
        i64::try_from(self.n)
            .map_err(|_| ModelError("basis template width exceeds tensor index range".into()))?;
        Ok(())
    }
}
