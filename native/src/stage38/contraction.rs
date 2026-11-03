//! Exact factor contraction and temporal-history recovery for stage 38.

mod count;
mod factorization;
mod ordering;
mod recover;
mod tables;

pub(super) use count::count_histories;
pub(super) use recover::recover_histories;
