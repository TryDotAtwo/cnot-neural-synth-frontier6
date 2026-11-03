use super::certificate::EvaluationError;
use super::evaluation::{evaluate, Stage26Values};
use super::CACHE_CAPACITY;
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureValue};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::FeatureSpec;
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub(super) struct SharedStage26 {
    pub(super) max_n: u32,
    pub(super) max_checker_pairs: usize,
    pub(super) cache: Mutex<VecDeque<(StateIdentity, Arc<Stage26Values>)>>,
}

impl SharedStage26 {
    fn values(&self, state: &MatrixState) -> Result<Arc<Stage26Values>, EvaluationError> {
        let identity = StateIdentity::from(state);
        if let Ok(cache) = self.cache.lock() {
            if let Some((_, values)) = cache.iter().find(|(key, _)| *key == identity) {
                return Ok(Arc::clone(values));
            }
        } else {
            return Err(EvaluationError::VerificationFailed(
                "stage26 shared cache mutex poisoned",
            ));
        }
        let values = Arc::new(evaluate(state, self.max_n, self.max_checker_pairs)?);
        let mut cache = self.cache.lock().map_err(|_| {
            EvaluationError::VerificationFailed("stage26 shared cache mutex poisoned")
        })?;
        if let Some(position) = cache.iter().position(|(key, _)| *key == identity) {
            cache.remove(position);
        }
        cache.push_front((identity, Arc::clone(&values)));
        while cache.len() > CACHE_CAPACITY {
            cache.pop_back();
        }
        Ok(values)
    }
}

pub(super) struct Stage26Producer {
    pub(super) spec: FeatureSpec,
    pub(super) shared: Arc<SharedStage26>,
}

impl FeatureProducer for Stage26Producer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, _policy: &FeaturePolicy) -> FeatureComputation {
        let values = match self.shared.values(state) {
            Ok(values) => values,
            Err(EvaluationError::UnsupportedAlphabet) => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "stage26_requires_full_unit_alphabet",
                );
            }
            Err(EvaluationError::ResourceLimit(reason)) => {
                return unavailable(FeatureStatus::ResourceLimit, reason);
            }
            Err(EvaluationError::VerificationFailed(reason)) => {
                panic!("{reason}");
            }
        };
        let (shape, payload) = match self.spec.id.ordinal() {
            3 => (vec![5], values.f03.clone()),
            4 => {
                let n = state.dimension() as usize;
                (vec![n, n, 3], values.f04.clone())
            }
            5 => (vec![2], values.f05.clone()),
            _ => unreachable!("stage26 registered only F03-F05"),
        };
        FeatureComputation::Computed(FeatureValue {
            shape,
            values: payload,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
