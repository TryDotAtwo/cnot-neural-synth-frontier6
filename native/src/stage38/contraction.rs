//! Exact factor contraction and temporal-history recovery for stage 38.

mod count;
mod factorization;
mod ordering;
mod recover;
mod tables;

#[cfg(all(test, feature = "cuda"))]
mod cuda_structure_tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_count_tests;

pub(super) use count::count_histories;
pub(super) use count::{count_histories_with_budget, HistoryCountBudget};
pub(super) use recover::recover_histories;
pub(super) use recover::recover_histories_with_limits;
