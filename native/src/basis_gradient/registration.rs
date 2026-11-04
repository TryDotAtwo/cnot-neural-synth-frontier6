//! Schema registration for E01, E04, and E05.

use super::{BasisGradientKind, BasisGradientProducer};
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureSpec};

pub fn register_basis_gradient(registry: &mut FeatureRegistry) -> Result<(), FeatureBuildError> {
    let ids = [
        FeatureId::new(b'E', 1).unwrap(),
        FeatureId::new(b'E', 4).unwrap(),
        FeatureId::new(b'E', 5).unwrap(),
    ];
    for id in ids {
        if registry.specs()[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let registrations = [
        (
            ids[0],
            BasisGradientKind::Cost,
            "partial_J17_zero_v1",
            "exact_integer_count",
        ),
        (
            ids[1],
            BasisGradientKind::PairGradient,
            "partial_J17_zero_gradient_v1",
            "float32_rounded_from_exact_counts",
        ),
        (
            ids[2],
            BasisGradientKind::GradientStatistics,
            "partial_J17_gradient_stats_v1",
            "float32_rounded_from_exact_counts",
        ),
    ];
    for (id, kind, producer_version, exactness) in registrations {
        let mut spec: FeatureSpec = feature_spec(id);
        spec.scope = "selected_partial_elimination;augmentation_only".into();
        spec.producer_version = producer_version.into();
        spec.dependencies = vec!["M".into(), "deterministic_partial_shear".into()];
        spec.exactness = exactness.into();
        spec.cost_tier = "T1".into();
        spec.wire_roles.clear();
        registry.register(Box::new(BasisGradientProducer { spec, kind }))?;
    }
    Ok(())
}
