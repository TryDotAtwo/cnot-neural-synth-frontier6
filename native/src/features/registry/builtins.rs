use super::FeatureRegistry;
use crate::features::{FeatureBuildError, FeatureProviderContexts};

impl FeatureRegistry {
    /// Register all 102 currently defined model-input features.
    ///
    /// The context bundle is explicit because these providers depend on physical
    /// cuts, bases, templates, alphabets, or experiment horizons. H07 is
    /// intentionally not registered: it is currently a post-forward diagnostic
    /// and has no model-input producer contract. Consequently, complete-103
    /// selection still fails closed on H07 alone.
    pub fn with_all_model_input_providers(
        contexts: FeatureProviderContexts,
    ) -> Result<Self, FeatureBuildError> {
        let FeatureProviderContexts {
            max_n,
            labeled_cuts,
            restricted_intersections,
            rank_increment_probes,
            alphabet,
            selected_residual_gates,
            block_parity_contexts,
            max_parity_patterns,
            max_parity_pairs,
            matrix_boundaries,
            max_boundary_ports,
            max_boundary_equations,
            paid_basis_contexts,
            paid_shear_config,
            template_response,
            bilinear_contexts,
            shared_relation_contexts,
            frequency_contexts,
            positive_packet_contexts,
            i09_contexts,
            i09_limits,
            i10_config,
        } = contexts;

        let mut registry = Self::with_default_context_free_providers(max_n)?;
        crate::rank_features::register_labeled_cuts(&mut registry, labeled_cuts)?;
        crate::rank_features::register_restricted_intersections(
            &mut registry,
            restricted_intersections,
        )?;
        crate::rank_features::register_rank_increment_probes(&mut registry, rank_increment_probes)?;
        crate::rank_features::register_selected_residual_ranks(
            &mut registry,
            &alphabet,
            selected_residual_gates,
        )?;
        crate::block_parities::register_block_parities(
            &mut registry,
            block_parity_contexts,
            max_parity_patterns,
            max_parity_pairs,
        )?;
        crate::matrix_boundary::register_matrix_boundaries(
            &mut registry,
            matrix_boundaries,
            max_boundary_ports,
            max_boundary_equations,
        )
        .map_err(|error| FeatureBuildError(error.to_string()))?;
        crate::paid_basis_features::register_paid_basis_features(
            &mut registry,
            paid_basis_contexts,
            paid_shear_config,
        )?;
        crate::template_response::register_template_response(&mut registry, template_response)?;
        crate::bilinear::register_default_bilinear_contexts(&mut registry, bilinear_contexts)?;
        crate::shared_relation::register_default_shared_relation_contexts(
            &mut registry,
            shared_relation_contexts,
        )?;
        crate::frequency_context::register_default_frequency_contexts(
            &mut registry,
            frequency_contexts,
        )?;
        crate::positive_packet::register_default_positive_packet_contexts(
            &mut registry,
            positive_packet_contexts,
        )?;
        crate::i09_quadratic::register_i09_quadratic_response(
            &mut registry,
            i09_contexts,
            i09_limits,
        )?;
        crate::response::register_tiny_i10(&mut registry, i10_config)?;
        Ok(registry)
    }

    /// Build a registry with default providers that need no experiment-specific
    /// physical contexts. max_n bounds the guarded recognizers and diffusion
    /// profile; their resource budgets and numeric settings are versioned by the
    /// providers. Template-response features and I10 remain opt-in because their
    /// exact experiment set and horizon are part of the feature identity.
    ///
    /// Context-bound providers and H07 are intentionally absent. H07 remains a
    /// post-forward diagnostic helper without a model-input producer contract.
    pub fn with_default_context_free_providers(max_n: u32) -> Result<Self, FeatureBuildError> {
        let mut registry = Self::new();
        crate::rank_features::register_partial_shear_rank(&mut registry)?;
        crate::basis_gradient::register_basis_gradient(&mut registry)?;
        crate::stage26::register_stage26_features_with_budget(&mut registry, max_n, 100_000)?;
        crate::bound_features::register_general_compiler_bounds(&mut registry)?;
        crate::guarded_features::register_guarded_pair_f02_view(&mut registry, max_n, 100_000)?;
        crate::guarded_features::register_guarded_pair(&mut registry, max_n, 100_000)?;
        crate::guarded_features::register_guarded_zero_excess(&mut registry, max_n)?;
        crate::guarded_features::register_guarded_one_excess(&mut registry, max_n, 100_000)?;
        let diffusion = crate::diffusion::DiffusionConfig {
            max_n,
            ..Default::default()
        };
        crate::diffusion::register_diffusion(&mut registry, diffusion)?;
        crate::symmetry_alignment::register_symmetry_alignment_h06(&mut registry)?;
        Ok(registry)
    }
}
