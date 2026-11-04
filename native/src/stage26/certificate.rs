mod construction;
mod gf2;
mod verification;

use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Certificate {
    pub(super) rows: Vec<BigUint>,
    pub(super) retained: Vec<bool>,
    pub(super) projected_rows: Vec<BigUint>,
    pub(super) deleted_columns: BTreeSet<usize>,
    pub(super) deletion_witness_rows: BTreeMap<usize, usize>,
    pub(super) targets: Vec<BigUint>,
    pub(super) h_basis: Vec<BigUint>,
    pub(super) deletions: usize,
    pub(super) target_count: usize,
    pub(super) obstruction_rank: usize,
    pub(super) h_dimension: usize,
    pub(super) w0_dimension: usize,
    pub(super) lower_bound: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum EvaluationError {
    UnsupportedAlphabet,
    ResourceLimit(&'static str),
    VerificationFailed(&'static str),
}

pub(super) fn make_certificate(state: &MatrixState) -> Certificate {
    construction::make_certificate(state)
}

pub(super) fn independently_verify(state: &MatrixState, certificate: &Certificate) -> bool {
    verification::independently_verify(state, certificate)
}

pub(super) fn cheap_lower_bound(rows: &[BigUint]) -> usize {
    construction::cheap_lower_bound(rows)
}

pub(super) fn high_remainder(value: BigUint, basis: &[BigUint]) -> BigUint {
    gf2::high_remainder(value, basis)
}
