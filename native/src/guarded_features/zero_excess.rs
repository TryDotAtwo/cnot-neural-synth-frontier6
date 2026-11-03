use super::{guarded_producer_version, unavailable, CACHE_CAPACITY, DEFAULT_MAX_ENTRIES};
use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::stage37::{recognize_stage37, verify_stage37_result, Stage37Error, Stage37Result};
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::Mutex;

struct CachedRecognition {
    state: MatrixState,
    result: Stage37Result,
}

struct GuardedZeroExcessProducer {
    spec: FeatureSpec,
    max_n: u32,
    cache: Mutex<VecDeque<CachedRecognition>>,
}

/// Registers F07 with the source's default one-excess contraction budget.
pub fn register_guarded_zero_excess(
    registry: &mut FeatureRegistry,
    max_n: u32,
) -> Result<(), FeatureBuildError> {
    register_guarded_zero_excess_with_budget(registry, max_n, DEFAULT_MAX_ENTRIES)
}

/// Registers F07 while preserving the source bundle's producer identity.
///
/// `max_entries` does not change F07's computation; the source uses a shared
/// producer version for F02/F06/F07/F08, so it is still part of the version
/// identity for checkpoint and packet compatibility.
pub fn register_guarded_zero_excess_with_budget(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F07 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 7).expect("F07 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 3;
    spec.scope = "solver_assisted;verified_zero_excess_scope".into();
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

    registry.register(Box::new(GuardedZeroExcessProducer {
        spec,
        max_n,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

impl FeatureProducer for GuardedZeroExcessProducer {
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
                "f07_configured_width_limit_exceeded",
            );
        }

        let alphabet = match CnotAlphabet::all_to_all_unit(state.dimension()) {
            Ok(alphabet) => alphabet,
            Err(_) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "f07_complete_alphabet_construction_failed",
                );
            }
        };
        if state.alphabet_id() != alphabet.id() {
            return unavailable(
                FeatureStatus::Unsupported,
                "f07_requires_complete_unit_all_to_all_cnot_alphabet",
            );
        }

        let result = match self.recognize(state, &alphabet) {
            Ok(result) => result,
            Err(Stage37Error::AlphabetMismatch) => {
                return unavailable(FeatureStatus::Unsupported, "f07_state_alphabet_mismatch");
            }
            Err(Stage37Error::IndexOverflow | Stage37Error::SizeOverflow) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "f07_recognizer_size_limit_exceeded",
                );
            }
            Err(error) => panic!("F07 exact recognizer integrity failure: {error}"),
        };

        let values = match &result {
            Stage37Result::Exact(_) => match verify_stage37_result(&result, &alphabet) {
                Ok(true) => vec![1.0, 0.0, 0.0],
                Ok(false) => panic!("F07 exact result failed independent admission"),
                Err(error) => panic!("F07 exact result verification failed: {error}"),
            },
            Stage37Result::NoZeroExcess(_) => vec![0.0, 1.0, 0.0],
            Stage37Result::Unsupported(_) => vec![0.0, 0.0, 1.0],
        };
        FeatureComputation::Computed(FeatureValue {
            shape: vec![3],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl GuardedZeroExcessProducer {
    fn recognize(
        &self,
        state: &MatrixState,
        alphabet: &CnotAlphabet,
    ) -> Result<Stage37Result, Stage37Error> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache.iter().position(|entry| entry.state == *state) {
                let entry = cache.remove(index).expect("cache index was found");
                let result = entry.result.clone();
                cache.push_front(entry);
                return Ok(result);
            }
        }

        let result = recognize_stage37(state, alphabet)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CachedRecognition {
                state: state.clone(),
                result: result.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        Ok(result)
    }
}
