//! Bounded caches for exact and Monte Carlo diffusion profiles.

mod mc;

use self::mc::{JointMoments, McKey};
use super::profiles::{ColumnProfile, PairProfile};
use super::{DiffusionConfig, MAX_CACHED_WIDTHS};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(super) struct ProfileCache {
    pub(super) config: DiffusionConfig,
    profiles: Mutex<HashMap<usize, Arc<ColumnProfile>>>,
    pair_profiles: Mutex<HashMap<usize, Arc<PairProfile>>>,
    mc_profiles: Mutex<HashMap<McKey, Arc<JointMoments>>>,
}

impl ProfileCache {
    pub(super) fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            profiles: Mutex::new(HashMap::new()),
            pair_profiles: Mutex::new(HashMap::new()),
            mc_profiles: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn get(&self, n: usize) -> Option<Arc<ColumnProfile>> {
        {
            let profiles = self
                .profiles
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(profile) = profiles.get(&n) {
                return Some(Arc::clone(profile));
            }
        }

        let profile = Arc::new(ColumnProfile::build(n, self.config)?);
        let mut profiles = self
            .profiles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = profiles.get(&n) {
            return Some(Arc::clone(existing));
        }
        if profiles.len() >= MAX_CACHED_WIDTHS {
            profiles.clear();
        }
        profiles.insert(n, Arc::clone(&profile));
        Some(profile)
    }

    pub(super) fn get_pair(&self, n: usize) -> Option<Arc<PairProfile>> {
        {
            let profiles = self
                .pair_profiles
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(profile) = profiles.get(&n) {
                return Some(Arc::clone(profile));
            }
        }

        let profile = Arc::new(PairProfile::build(n, self.config)?);
        let mut profiles = self
            .pair_profiles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = profiles.get(&n) {
            return Some(Arc::clone(existing));
        }
        if profiles.len() >= MAX_CACHED_WIDTHS / 2 {
            profiles.clear();
        }
        profiles.insert(n, Arc::clone(&profile));
        Some(profile)
    }
}
