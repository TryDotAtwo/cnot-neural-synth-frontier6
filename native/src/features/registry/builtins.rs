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
        let max_n = contexts.max_n;
        Self::with_all_model_input_providers_f08_limit(contexts, max_n, None, None, None)
    }

    pub(crate) fn with_all_model_input_providers_f08_limit(
        contexts: FeatureProviderContexts,
        f08_max_n: u32,
        f08_max_changed_rows: Option<u32>,
        f08_max_component_events: Option<u32>,
        f08_max_history_counts: Option<u32>,
    ) -> Result<Self, FeatureBuildError> {
        if f08_max_n < 2 || f08_max_n > contexts.max_n {
            return Err(FeatureBuildError(
                "F08 width limit must be between two and max_n".into(),
            ));
        }
        let FeatureProviderContexts {
            max_n,
            labeled_cuts,
            restricted_intersections,
            rank_increment_probes,
            alphabet,
            selected_residual_gates,
            selected_residual_additional_contexts,
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

        let mut registry = Self::with_context_free_providers_f08_limit(max_n, f08_max_n, f08_max_changed_rows, f08_max_component_events, f08_max_history_counts)?;
        crate::rank_features::register_labeled_cuts(&mut registry, labeled_cuts)?;
        crate::rank_features::register_restricted_intersections(
            &mut registry,
            restricted_intersections,
        )?;
        crate::rank_features::register_rank_increment_probes(&mut registry, rank_increment_probes)?;
        if selected_residual_additional_contexts.is_empty() {
            crate::rank_features::register_selected_residual_ranks(
                &mut registry, &alphabet, selected_residual_gates,
            )?;
        } else {
            let mut profiles = Vec::with_capacity(selected_residual_additional_contexts.len() + 1);
            profiles.push(crate::rank_features::SelectedResidualContext {
                alphabet,
                gates: selected_residual_gates,
            });
            profiles.extend(selected_residual_additional_contexts);
            crate::rank_features::register_selected_residual_ranks_multi(&mut registry, profiles)?;
        }
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
        Self::with_context_free_providers_f08_limit(max_n, max_n, None, None, None)
    }

    fn with_context_free_providers_f08_limit(
        max_n: u32,
        f08_max_n: u32,
        f08_max_changed_rows: Option<u32>,
        f08_max_component_events: Option<u32>,
        f08_max_history_counts: Option<u32>,
    ) -> Result<Self, FeatureBuildError> {
        let mut registry = Self::new();
        crate::rank_features::register_partial_shear_rank(&mut registry)?;
        crate::basis_gradient::register_basis_gradient(&mut registry)?;
        crate::stage26::register_stage26_features_with_budget(&mut registry, max_n, 100_000)?;
        crate::bound_features::register_general_compiler_bounds(&mut registry)?;
        crate::guarded_features::register_guarded_pair_f02_view(&mut registry, max_n, 100_000)?;
        crate::guarded_features::register_guarded_pair(&mut registry, max_n, 100_000)?;
        crate::guarded_features::register_guarded_zero_excess(&mut registry, max_n)?;
        crate::guarded_features::register_guarded_one_excess_with_limits(
            &mut registry, f08_max_n, 100_000, f08_max_changed_rows, f08_max_component_events, f08_max_history_counts,
        )?;
        let diffusion = crate::diffusion::DiffusionConfig {
            max_n,
            ..Default::default()
        };
        crate::diffusion::register_diffusion(&mut registry, diffusion)?;
        crate::symmetry_alignment::register_symmetry_alignment_h06(&mut registry)?;
        Ok(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::FeatureRegistry;
    use crate::{CnotAlphabet, FeatureId, FeaturePolicy, FeatureStatus, MatrixState};

    #[test]
    fn independent_f08_limit_changes_only_f08_contract_and_status() {
        let ordinary = FeatureRegistry::with_default_context_free_providers(3).unwrap();
        let limited = FeatureRegistry::with_context_free_providers_f08_limit(3, 2, None, None, None).unwrap();
        let f08 = FeatureId::new(b'F', 8).unwrap();
        for (before, after) in ordinary.specs().iter().zip(limited.specs()) {
            if before.id == f08 {
                assert_ne!(before.producer_version, after.producer_version);
            } else {
                assert_eq!(*before, after, "{} contract changed", before.id);
            }
        }
        assert_ne!(ordinary.contract_sha256(), limited.contract_sha256());

        let mut policy = FeaturePolicy::all_model_inputs_augmentation_only(
            crate::schema::MISSING_TOKEN_LAYOUT_ID,
        );
        policy.enabled_ids = [2, 6, 7, 8]
            .into_iter()
            .map(|ordinal| FeatureId::new(b'F', ordinal).unwrap())
            .collect();
        policy.solver_assisted = true;
        let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
        let state = MatrixState::identity(3, alphabet.id()).unwrap();
        let baseline = ordinary.build(&[state.clone()], &policy).unwrap();
        let capped = limited.build(&[state], &policy).unwrap();
        assert_eq!(baseline.feature(f08).status, vec![FeatureStatus::Computed]);
        assert_eq!(
            capped.feature(f08).status,
            vec![FeatureStatus::ResourceLimit]
        );
        for ordinal in [2, 6, 7] {
            let id = FeatureId::new(b'F', ordinal).unwrap();
            assert_eq!(
                baseline.feature(id).status,
                capped.feature(id).status,
                "{id}"
            );
            assert_eq!(
                baseline.feature(id).values,
                capped.feature(id).values,
                "{id}"
            );
        }
    }
}
