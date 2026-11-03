use super::{computed_status, guarded_producer_version, unavailable, CACHE_CAPACITY};
use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::stage38::{recognize_stage38, verify_stage38_result, Stage38Error, Stage38Result};
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
    cache: Mutex<VecDeque<CachedOneExcess>>,
}

/// Registers source F08 with explicit width and exact-contraction budgets.
pub fn register_guarded_one_excess(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
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

    let id = FeatureId::new(b'F', 8).expect("F08 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 3;
    spec.scope = "solver_assisted;verified_one_excess_scope;resource_limit_unknown".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
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

        let result = recognize_stage38(state, alphabet, self.max_entries)?;
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
