mod digest;

use super::super::{DEFAULT_MAX_DIMENSION, DEFAULT_MAX_MATRIX_BITS};
use super::BilinearProducer;
use crate::bilinear::BilinearContext;
use crate::features::{FeatureBuildError, FeatureRegistry};
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use std::collections::HashSet;

/// Registers I05 for one or more explicit labelled matrix widths.
pub fn register_bilinear_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<BilinearContext>,
    max_dimension: usize,
    max_matrix_bits: usize,
) -> Result<(), FeatureBuildError> {
    if contexts.is_empty() || max_dimension == 0 || max_matrix_bits == 0 {
        return Err(FeatureBuildError(
            "I05 requires explicit contexts and positive dimension/matrix-bit budgets".into(),
        ));
    }
    let mut widths = HashSet::with_capacity(contexts.len());
    if contexts.iter().any(|context| !widths.insert(context.n)) {
        return Err(FeatureBuildError(
            "I05 allows one labelled port context per physical width".into(),
        ));
    }

    let producer_version = digest::producer_version(&contexts, max_dimension, max_matrix_bits);
    let id = FeatureId::new(b'I', 5).expect("I05 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 2;
    let spec = spec.with_provider_metadata(
        crate::schema::ProviderMetadata::new(
            concat!(
                "explicit_stage55_UA=A0_DV=D0_DA=C_matrix_product_fiber;",
                "C=current_labelled_physical_block;residual_rank_and_free_hidden_width;",
                "augmentation_only;not_CNOT_distance"
            ),
            producer_version,
            "exact",
            "T2",
        )
        .dependencies(["M", "physical_block", "independent_U_V", "labelled_A0_D0"])
        .wire_roles(std::iter::empty::<String>()),
    );
    registry.register(Box::new(BilinearProducer {
        spec,
        contexts,
        max_dimension,
        max_matrix_bits,
    }))
}

pub fn register_default_bilinear_contexts(
    registry: &mut FeatureRegistry,
    contexts: Vec<BilinearContext>,
) -> Result<(), FeatureBuildError> {
    register_bilinear_contexts(
        registry,
        contexts,
        DEFAULT_MAX_DIMENSION,
        DEFAULT_MAX_MATRIX_BITS,
    )
}
