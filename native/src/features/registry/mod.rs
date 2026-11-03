use super::FeatureProducer;
use crate::schema::{all_feature_specs, feature_spec, FeatureId, FeatureSpec, FEATURE_COUNT};
use crate::structural;
use std::collections::HashMap;

mod builtins;
mod contract;
mod registration;
mod selection;

pub struct FeatureRegistry {
    pub(super) specs: HashMap<FeatureId, FeatureSpec>,
    pub(super) providers: HashMap<FeatureId, Box<dyn FeatureProducer>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut specs = all_feature_specs();
        for spec in &mut specs {
            if let Some(metadata) = structural::metadata(spec.id) {
                spec.scope = metadata.scope.into();
                spec.producer_version = metadata.producer_version.into();
                spec.dependencies = metadata
                    .dependencies
                    .iter()
                    .map(|value| (*value).into())
                    .collect();
                spec.exactness = metadata.exactness.into();
                spec.cost_tier = metadata.cost_tier.into();
                spec.wire_roles.clear();
            }
        }
        Self {
            specs: specs.into_iter().map(|spec| (spec.id, spec)).collect(),
            providers: HashMap::new(),
        }
    }

    /// Returns specs in the canonical 103-ID order, including the actual
    /// producer metadata used by this registry.
    pub fn specs(&self) -> Vec<FeatureSpec> {
        (0..FEATURE_COUNT)
            .filter_map(FeatureId::from_index)
            .map(|id| {
                self.specs
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| feature_spec(id))
            })
            .collect()
    }
}
