//! Bounded LRU cache for deterministic stage-35 recognition results.

use super::super::CACHE_CAPACITY;
use super::{CachedPair, GuardedPairProducer};
use crate::alphabet::CnotAlphabet;
use crate::stage35::{recognize_stage35, Stage35Error, Stage35Result};
use crate::state::MatrixState;

impl GuardedPairProducer {
    pub(super) fn recognize(
        &self,
        state: &MatrixState,
        alphabet: &CnotAlphabet,
    ) -> Result<Stage35Result, Stage35Error> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache.iter().position(|entry| entry.state == *state) {
                let entry = cache.remove(index).expect("cache index was found");
                let result = entry.result.clone();
                cache.push_front(entry);
                return Ok(result);
            }
        }

        let result = recognize_stage35(state, alphabet)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CachedPair {
                state: state.clone(),
                result: result.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        Ok(result)
    }
}
