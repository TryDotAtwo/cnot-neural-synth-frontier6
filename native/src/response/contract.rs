//! Stable limits and result types for the exact finite-horizon I10 oracle.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct I10Config {
    pub max_n: u32,
    pub max_degree: usize,
    pub max_layer_states: usize,
    pub max_transitions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum I10OracleStatus {
    Exact,
    Unsupported,
    ResourceLimit,
}

/// Direct oracle result. `positive_terms` is empty on non-exact outcomes and
/// otherwise contains `(physical_degree, exact_word_count)` in degree order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct I10OracleResult {
    pub status: I10OracleStatus,
    pub positive_terms: Vec<(usize, u128)>,
    pub evaluations: usize,
    pub reason: String,
    pub scope: String,
}
