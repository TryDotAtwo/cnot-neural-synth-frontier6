use super::{computed_status, guarded_producer_version, unavailable, CACHE_CAPACITY};
use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::stage38::{recognize_stage38_with_limits, verify_stage38_result, Stage38Error, Stage38Result};
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::Mutex;

struct CachedOneExcess {
    state: MatrixState,
    result: Stage38Result,
}

struct GuardedOneExcessProducer {
    spec: FeatureSpec,
    max_n: u32,
    max_entries: usize,
    max_changed_rows: Option<u32>,
    max_component_events: Option<u32>,
    max_history_counts: Option<u32>,
    cache: Mutex<VecDeque<CachedOneExcess>>,
}

/// Registers source F08 with explicit width and exact-contraction budgets.
pub fn register_guarded_one_excess(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    register_guarded_one_excess_with_changed_limit(registry, max_n, max_entries, None)
}

/// Limit costly one-excess enumeration by the number of nonidentity rows.
/// Zero-excess results are still computed even if they exceed the limit.
pub fn register_guarded_one_excess_with_changed_limit(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
    max_changed_rows: Option<u32>,
) -> Result<(), FeatureBuildError> {
    register_guarded_one_excess_with_limits(registry, max_n, max_entries, max_changed_rows, None, None)
}

pub(crate) fn register_guarded_one_excess_with_limits(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
    max_changed_rows: Option<u32>,
    max_component_events: Option<u32>,
    max_history_counts: Option<u32>,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F08 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }
    if max_changed_rows.is_some_and(|limit| limit == 0 || limit > max_n) {
        return Err(FeatureBuildError("F08 changed-row limit must be between one and max_n".into()));
    }
    if max_component_events.is_some_and(|limit| limit == 0 || u64::from(limit) > u64::from(max_n) + 1) {
        return Err(FeatureBuildError("F08 component limit must be between one and max_n plus one".into()));
    }
    if max_history_counts == Some(0) {
        return Err(FeatureBuildError("F08 history count limit must be positive".into()));
    }

    let id = FeatureId::new(b'F', 8).expect("F08 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 3;
    spec.scope = "solver_assisted;verified_one_excess_scope;resource_limit_unknown".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    if let Some(limit) = max_changed_rows {
        spec.producer_version.push_str(&format!(":changed_rows_no_zero_v1:{limit}"));
    }
    if let Some(limit) = max_component_events {
        spec.producer_version.push_str(&format!(":component_events_v1:{limit}"));
    }
    if let Some(limit) = max_history_counts {
        spec.producer_version.push_str(&format!(":history_counts_v1:{limit}"));
    }
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    spec.wire_roles.clear();
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedOneExcessProducer {
        spec,
        max_n,
        max_entries,
        max_changed_rows,
        max_component_events,
        max_history_counts,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

impl FeatureProducer for GuardedOneExcessProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if !policy.solver_assisted {
            return unavailable(
                FeatureStatus::Unsupported,
                "guarded_recognizer_requires_solver_assisted_policy",
            );
        }
        if state.dimension() > self.max_n {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "f08_configured_width_limit_exceeded",
            );
        }
        let alphabet = match CnotAlphabet::all_to_all_unit(state.dimension()) {
            Ok(alphabet) => alphabet,
            Err(_) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "f08_complete_alphabet_construction_failed",
                );
            }
        };
        if state.alphabet_id() != alphabet.id() {
            return unavailable(
                FeatureStatus::Unsupported,
                "f08_requires_complete_unit_all_to_all_cnot_alphabet",
            );
        }

        if let Some(limit) = self.max_changed_rows {
            let changed = state.packed_rows().chunks_exact(state.words_per_row())
                .enumerate().filter(|(row, words)| words.iter().enumerate().any(|(word, &bits)| {
                    let identity = if *row / 64 == word { 1u64 << (*row % 64) } else { 0 };
                    bits != identity
                })).count();
            if changed > limit as usize {
                match crate::stage37::recognize_stage37(state, &alphabet) {
                    Ok(crate::stage37::Stage37Result::NoZeroExcess(_)) => {
                        return unavailable(FeatureStatus::ResourceLimit, "f08_changed_row_limit_exceeded");
                    }
                    Ok(_) => {}
                    Err(error) => panic!("F08 zero-excess prerequisite failure: {error}"),
                }
            }
        }

        let result = match self.recognize(state, &alphabet) {
            Ok(result) => result,
            Err(Stage38Error::AlphabetMismatch) => {
                return unavailable(FeatureStatus::Unsupported, "f08_state_alphabet_mismatch");
            }
            Err(Stage38Error::InvalidBudget | Stage38Error::SizeOverflow) => {
                return unavailable(FeatureStatus::ResourceLimit, "f08_resource_limit");
            }
            Err(error) => panic!("F08 exact recognizer integrity failure: {error}"),
        };

        match &result {
            Stage38Result::Exact(_) => match verify_stage38_result(&result, &alphabet) {
                Ok(true) => computed_status(vec![1.0, 0.0, 0.0]),
                Ok(false) => panic!("F08 exact result failed independent admission"),
                Err(error) => panic!("F08 exact result verification failed: {error}"),
            },
            Stage38Result::CertifiedNoSolutionWithinScope(_) => {
                computed_status(vec![0.0, 1.0, 0.0])
            }
            Stage38Result::Unsupported(_) => computed_status(vec![0.0, 0.0, 1.0]),
            Stage38Result::ResourceLimit(limit) => {
                unavailable(FeatureStatus::ResourceLimit, &limit.reason)
            }
        }
    }
}

impl GuardedOneExcessProducer {
    fn recognize(
        &self,
        state: &MatrixState,
        alphabet: &CnotAlphabet,
    ) -> Result<Stage38Result, Stage38Error> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache.iter().position(|entry| entry.state == *state) {
                let entry = cache.remove(index).expect("cache index was found");
                let result = entry.result.clone();
                cache.push_front(entry);
                return Ok(result);
            }
        }

        let result = recognize_stage38_with_limits(
            state, alphabet, self.max_entries, self.max_component_events.map(|v| v as usize), self.max_history_counts,
        )?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CachedOneExcess {
                state: state.clone(),
                result: result.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        Ok(result)
    }
}
