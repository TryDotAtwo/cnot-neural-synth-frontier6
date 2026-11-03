//! Shared bounded cache and feature-producer adapter for D09-D14.
use super::compute::compute_paid_basis_features;
use super::helpers::unavailable;
use super::values::PaidBasisFeatureValues;
use super::{CACHE_CAPACITY, PaidBasisBlockContext};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::paid_basis::PaidShearConfig;
use crate::schema::FeatureId;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub(super) struct PaidBasisFeatureProducer {
    pub(super) id: FeatureId,
    pub(super) spec: FeatureSpec,
    pub(super) shared: Arc<PaidBasisFeatureCache>,
}

pub(super) struct PaidBasisFeatureCache {
    contexts: Vec<PaidBasisBlockContext>,
    config: PaidShearConfig,
    entries: Mutex<VecDeque<(StateIdentity, CachedPaidBasisFeatures)>>,
}

#[derive(Clone)]
pub(super) enum CachedPaidBasisFeatures {
    Computed(PaidBasisFeatureValues),
    Unavailable(FeatureStatus, String),
}

impl PaidBasisFeatureValues {
    fn for_id(&self, id: FeatureId) -> FeatureComputation {
        match id.ordinal() {
            9 => FeatureComputation::Computed(self.d09.clone()),
            10 => FeatureComputation::Computed(self.d10.clone()),
            11 => FeatureComputation::Computed(self.d11.clone()),
            12 => FeatureComputation::Computed(self.d12.clone()),
            13 => FeatureComputation::Computed(self.d13.clone()),
            14 => FeatureComputation::ComputedWithProvenance {
                value: self.d14.clone(),
                provenance: self.d14_provenance.clone(),
            },
            _ => unreachable!("only D09-D14 share this producer cache"),
        }
    }
}

impl PaidBasisFeatureCache {
    fn get(&self, state: &MatrixState, policy: &FeaturePolicy) -> CachedPaidBasisFeatures {
        let key = StateIdentity::from(state);
        let Ok(mut entries) = self.entries.lock() else {
            return CachedPaidBasisFeatures::Unavailable(
                FeatureStatus::ResourceLimit,
                "paid_basis_cache_poisoned".into(),
            );
        };
        if let Some(index) = entries.iter().position(|(cached, _)| cached == &key) {
            let entry = entries
                .remove(index)
                .expect("located paid-basis cache entry");
            let result = entry.1.clone();
            entries.push_back(entry);
            return result;
        }
        let result = compute_paid_basis_features(state, policy, &self.contexts, self.config);
        if entries.len() == CACHE_CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, result.clone()));
        result
    }
}

impl PaidBasisFeatureCache {
    pub(super) fn new(contexts: Vec<PaidBasisBlockContext>, config: PaidShearConfig) -> Self {
        Self {
            contexts,
            config,
            entries: Mutex::new(VecDeque::new()),
        }
    }
}

impl FeatureProducer for PaidBasisFeatureProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != self.symmetry_contract() {
            return unavailable(
                FeatureStatus::Unsupported,
                "paid_basis_requires_augmentation_only_symmetry",
            );
        }
        match self.shared.get(state, policy) {
            CachedPaidBasisFeatures::Computed(values) => values.for_id(self.id),
            CachedPaidBasisFeatures::Unavailable(status, reason) => unavailable(status, reason),
        }
    }
}
