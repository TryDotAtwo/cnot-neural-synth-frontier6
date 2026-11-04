//! Shared cache and feature-producer implementation for the response family.

use super::budget::template_response_workspace_bytes;
use super::compute::{compute_template_response, resource_limit, unavailable};
use super::{TemplateResponseConfig, TEMPLATE_RESPONSE_CACHE_CAPACITY};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureValue};
use crate::packet::{FeatureStatus, StateIdentity};
use crate::schema::{FeatureId, FeatureSpec};
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub(super) struct TemplateResponseProducer {
    id: FeatureId,
    spec: FeatureSpec,
    shared: Arc<TemplateResponseCache>,
}

pub(super) struct TemplateResponseCache {
    config: TemplateResponseConfig,
    entries: Mutex<VecDeque<(StateIdentity, CachedTemplateResponse)>>,
}

#[derive(Clone)]
pub(super) enum CachedTemplateResponse {
    Computed(Box<TemplateResponseValues>),
    Unavailable(FeatureStatus, String),
}

#[derive(Clone)]
pub(super) struct TemplateResponseValues {
    pub(super) e02: FeatureValue,
    pub(super) e03: FeatureValue,
    pub(super) e06: FeatureValue,
    pub(super) e07: FeatureValue,
    pub(super) e08: FeatureValue,
    pub(super) e09: FeatureValue,
}

impl TemplateResponseValues {
    fn for_id(&self, id: FeatureId) -> FeatureValue {
        match id.ordinal() {
            2 => self.e02.clone(),
            3 => self.e03.clone(),
            6 => self.e06.clone(),
            7 => self.e07.clone(),
            8 => self.e08.clone(),
            9 => self.e09.clone(),
            _ => unreachable!("only template response IDs share this provider"),
        }
    }
}

impl TemplateResponseCache {
    pub(super) fn new(config: TemplateResponseConfig) -> Self {
        Self {
            config,
            entries: Mutex::new(VecDeque::new()),
        }
    }

    fn get(&self, state: &MatrixState) -> CachedTemplateResponse {
        let key = StateIdentity::from(state);
        let Ok(mut entries) = self.entries.lock() else {
            return CachedTemplateResponse::Unavailable(
                FeatureStatus::ResourceLimit,
                "template_response_cache_poisoned".into(),
            );
        };
        if let Some(index) = entries.iter().position(|(cached, _)| cached == &key) {
            let entry = entries.remove(index).expect("located cache entry");
            let result = entry.1.clone();
            entries.push_back(entry);
            return result;
        }

        let result = compute_template_response(state, &self.config);
        if entries.len() == TEMPLATE_RESPONSE_CACHE_CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, result.clone()));
        result
    }
}

impl TemplateResponseProducer {
    pub(super) fn new(
        id: FeatureId,
        spec: FeatureSpec,
        shared: Arc<TemplateResponseCache>,
    ) -> Self {
        Self { id, spec, shared }
    }
}

impl FeatureProducer for TemplateResponseProducer {
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
                "template_response_requires_augmentation_only_symmetry",
            );
        }
        if policy.max_bytes.is_some_and(|limit| {
            template_response_workspace_bytes(state.dimension(), self.shared.config.choices.len())
                .is_none_or(|bytes| bytes > limit)
        }) {
            return resource_limit("template_response_workspace_budget_exhausted");
        }
        match self.shared.get(state) {
            CachedTemplateResponse::Computed(values) => {
                FeatureComputation::Computed(values.for_id(self.id))
            }
            CachedTemplateResponse::Unavailable(status, reason) => {
                FeatureComputation::Unavailable { status, reason }
            }
        }
    }
}
