mod compute;

use super::{MatrixBoundarySpec, CACHE_CAPACITY};
use crate::features::FeatureComputation;
use crate::packet::StateIdentity;
use crate::schema::FeatureId;
use crate::state::MatrixState;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

#[derive(Clone)]
struct BoundaryCacheEntry {
    state: StateIdentity,
    max_bytes: Option<usize>,
    values: HashMap<FeatureId, FeatureComputation>,
}

pub(super) struct MatrixBoundaryContext {
    specs: Vec<MatrixBoundarySpec>,
    max_ports: usize,
    max_equations: usize,
    cache: Mutex<VecDeque<BoundaryCacheEntry>>,
}

impl MatrixBoundaryContext {
    pub(super) fn new(
        specs: Vec<MatrixBoundarySpec>,
        max_ports: usize,
        max_equations: usize,
    ) -> Self {
        Self {
            specs,
            max_ports,
            max_equations,
            cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
        }
    }

    pub(super) fn values_for(
        &self,
        state: &MatrixState,
        max_bytes: Option<usize>,
    ) -> HashMap<FeatureId, FeatureComputation> {
        let identity = StateIdentity::from(state);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache
                .iter()
                .position(|entry| entry.state == identity && entry.max_bytes == max_bytes)
            {
                if let Some(entry) = cache.remove(index) {
                    let result = entry.values.clone();
                    cache.push_front(entry);
                    return result;
                }
            }
        }

        let values = self.compute(state, max_bytes);
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(BoundaryCacheEntry {
                state: identity,
                max_bytes,
                values: values.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        values
    }
}
