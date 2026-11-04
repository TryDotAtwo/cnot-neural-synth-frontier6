//! Canonical square-free Boolean polynomials over GF(2).

use crate::i09_quadratic::I09Monomial;
use std::collections::BTreeSet;

#[derive(Default)]
pub(super) struct Polynomial(BTreeSet<I09Monomial>);

impl Polynomial {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn toggle(&mut self, term: I09Monomial) {
        if !self.0.insert(term) {
            self.0.remove(&term);
        }
    }

    pub(super) fn finish(self) -> Vec<I09Monomial> {
        self.0.into_iter().collect()
    }
}

pub(super) fn product(left: usize, right: usize) -> I09Monomial {
    if left == right {
        I09Monomial::Variable(left)
    } else if left < right {
        I09Monomial::Product(left, right)
    } else {
        I09Monomial::Product(right, left)
    }
}

pub(super) fn monomial_is_in_range(term: I09Monomial, variable_count: usize) -> bool {
    match term {
        I09Monomial::Constant => true,
        I09Monomial::Variable(index) => index < variable_count,
        I09Monomial::Product(left, right) => left < variable_count && right < variable_count,
    }
}
