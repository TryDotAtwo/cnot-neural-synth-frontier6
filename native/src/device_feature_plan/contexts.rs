use super::block_histogram::BlockHistogramPlan;
use super::boundary::BoundaryPlan;
use crate::device_context::{
    DeviceBlockParityContext, DeviceLabeledCutContext, DeviceMatrixBoundaryContext,
    DeviceRankIncrementContext, DeviceRestrictedIntersectionContext, DeviceSelectedResidualContext,
};
use crate::{
    BlockContext, CnotAlphabet, CnotGate, FeaturePolicy, FeatureProviderContexts, FeatureRegistry,
    LabeledCut, MatrixBoundarySpec, RankIncrementProbe, RestrictedIntersection,
};

#[derive(Clone, Copy)]
pub struct SelectedResidualPlanInputs<'a> {
    pub alphabet: &'a CnotAlphabet,
    pub gates: &'a [CnotGate],
}

#[derive(Clone, Copy)]
pub struct BlockParityPlanInputs<'a> {
    pub contexts: &'a [BlockContext],
    pub max_patterns: usize,
    pub max_pairs: usize,
}

#[derive(Clone, Copy)]
pub struct MatrixBoundaryPlanInputs<'a> {
    pub boundaries: &'a [MatrixBoundarySpec],
    pub max_ports: usize,
    pub max_equations: usize,
}

#[derive(Clone, Copy)]
pub struct PaidBasisPlanInputs<'a> {
    pub contexts: &'a [crate::PaidBasisBlockContext],
    pub config: crate::PaidShearConfig,
}

/// Borrowed setup inputs. Only contexts for selected IDs are required. Lists
/// include all configured widths so producer fingerprints match the registry.
#[derive(Clone, Copy, Default)]
pub struct DeviceFeatureContextInputs<'a> {
    pub labeled_cuts: &'a [LabeledCut],
    pub restricted_intersections: &'a [RestrictedIntersection],
    pub rank_increment_probes: &'a [RankIncrementProbe],
    pub selected_residual: Option<SelectedResidualPlanInputs<'a>>,
    pub selected_residual_additional_contexts: &'a [crate::SelectedResidualContext],
    pub block_parities: Option<BlockParityPlanInputs<'a>>,
    pub matrix_boundaries: Option<MatrixBoundaryPlanInputs<'a>>,
    pub stage26: Option<super::Stage26PlanInputs>,
    pub frequency: Option<super::FrequencyPlanInputs<'a>>,
    pub shared_relation: Option<super::SharedRelationPlanInputs<'a>>,
    pub bilinear: Option<super::BilinearPlanInputs<'a>>,
    pub positive_packet: Option<super::PositivePacketPlanInputs<'a>>,
    pub quadratic: Option<super::QuadraticPlanInputs<'a>>,
    pub endpoint_response: Option<crate::I10Config>,
    pub paid_basis: Option<PaidBasisPlanInputs<'a>>,
    pub template_response: Option<&'a crate::TemplateResponseConfig>,
    pub guarded: Option<super::GuardedPlanInputs>,
    pub diffusion: Option<crate::DiffusionConfig>,
}

impl<'a> From<&'a FeatureProviderContexts> for DeviceFeatureContextInputs<'a> {
    fn from(contexts: &'a FeatureProviderContexts) -> Self {
        Self {
            labeled_cuts: &contexts.labeled_cuts,
            restricted_intersections: &contexts.restricted_intersections,
            rank_increment_probes: &contexts.rank_increment_probes,
            selected_residual: Some(SelectedResidualPlanInputs {
                alphabet: &contexts.alphabet,
                gates: &contexts.selected_residual_gates,
            }),
            selected_residual_additional_contexts: &contexts.selected_residual_additional_contexts,
            block_parities: Some(BlockParityPlanInputs {
                contexts: &contexts.block_parity_contexts,
                max_patterns: contexts.max_parity_patterns,
                max_pairs: contexts.max_parity_pairs,
            }),
            matrix_boundaries: Some(MatrixBoundaryPlanInputs {
                boundaries: &contexts.matrix_boundaries,
                max_ports: contexts.max_boundary_ports,
                max_equations: contexts.max_boundary_equations,
            }),
            stage26: Some(super::Stage26PlanInputs {
                max_n: contexts.max_n,
                max_checker_pairs: crate::stage26::DEFAULT_MAX_CHECKER_PAIRS,
            }),
            frequency: Some(super::FrequencyPlanInputs {
                contexts: &contexts.frequency_contexts,
                max_width: crate::frequency_context::DEFAULT_MAX_WIDTH,
                max_frequency_bits: crate::frequency_context::DEFAULT_MAX_FREQUENCY_BITS,
            }),
            shared_relation: Some(super::SharedRelationPlanInputs {
                contexts: &contexts.shared_relation_contexts,
                max_vectors: crate::shared_relation::DEFAULT_MAX_VECTORS,
                max_coordinates: crate::shared_relation::DEFAULT_MAX_COORDINATES,
                max_ports: crate::shared_relation::DEFAULT_MAX_PORTS,
            }),
            bilinear: Some(super::BilinearPlanInputs {
                contexts: &contexts.bilinear_contexts,
                max_dimension: crate::bilinear::DEFAULT_MAX_DIMENSION,
                max_matrix_bits: crate::bilinear::DEFAULT_MAX_MATRIX_BITS,
            }),
            positive_packet: Some(super::PositivePacketPlanInputs {
                contexts: &contexts.positive_packet_contexts,
                max_n: crate::positive_packet::DEFAULT_MAX_N,
                max_slots: crate::positive_packet::DEFAULT_MAX_SLOTS,
                max_monomials: crate::positive_packet::DEFAULT_MAX_MONOMIALS,
                max_factor_entries: crate::positive_packet::DEFAULT_MAX_FACTOR_ENTRIES,
            }),
            quadratic: Some(super::QuadraticPlanInputs {
                contexts: &contexts.i09_contexts,
                limits: contexts.i09_limits,
            }),
            endpoint_response: Some(contexts.i10_config),
            paid_basis: Some(PaidBasisPlanInputs {
                contexts: &contexts.paid_basis_contexts,
                config: contexts.paid_shear_config,
            }),
            template_response: Some(&contexts.template_response),
            diffusion: Some(crate::DiffusionConfig {
                max_n: contexts.max_n,
                ..Default::default()
            }),
            guarded: Some(super::GuardedPlanInputs {
                max_n: contexts.max_n,
                max_entries: 100_000,
            }),
        }
    }
}

pub(super) struct PreparedContexts {
    pub(super) unsupported_at_width: std::collections::HashSet<crate::FeatureId>,
    pub(super) dimension: u32,
    pub(super) cuts: Option<DeviceLabeledCutContext>,
    pub(super) partial_shear: bool,
    pub(super) paid_basis: Option<crate::device_paid_basis::DevicePaidBasisPlan>,
    pub(super) template_response:
        Option<crate::device_template_response::DeviceTemplateResponsePlan>,
    pub(super) basis_gradient: bool,
    pub(super) general_compiler: Option<crate::device_compiler::DeviceGeneralCompilerPlan>,
    pub(super) stage26: Option<super::stage26::Stage26Plan>,
    pub(super) guarded_pair: Option<crate::device_guarded_pair::DeviceGuardedPairPlan>,
    pub(super) zero_excess: Option<crate::device_zero_excess::DeviceZeroExcessPlan>,
    pub(super) one_excess: Option<crate::device_one_excess::DeviceOneExcessPlan>,
    pub(super) column_diffusion: Option<super::column_diffusion::ColumnDiffusion>,
    pub(super) pair_diffusion: Option<super::pair_diffusion::PairDiffusion>,
    pub(super) shared_word: Option<super::shared_word::SharedWord>,
    pub(super) frequency: Option<super::frequency::FrequencyPlan>,
    pub(super) shared_relation: Option<super::shared_relation::SharedRelationPlan>,
    pub(super) bilinear: Option<super::bilinear::BilinearPlan>,
    pub(super) positive_packet: Option<super::positive_packet::PositivePacketPlan>,
    pub(super) quadratic: Option<crate::device_quadratic::DeviceQuadraticPlan>,
    pub(super) endpoint_response:
        Option<crate::device_endpoint_response::DeviceEndpointResponsePlan>,
    pub(super) intersections: Option<DeviceRestrictedIntersectionContext>,
    pub(super) increments: Option<DeviceRankIncrementContext>,
    pub(super) residual: Option<DeviceSelectedResidualContext>,
    pub(super) blocks: Option<DeviceBlockParityContext>,
    pub(super) block_histogram: Option<BlockHistogramPlan>,
    pub(super) boundary: Option<DeviceMatrixBoundaryContext>,
    pub(super) boundary_limited: bool,
    pub(super) boundary_relation: Option<BoundaryPlan>,
}

impl PreparedContexts {
    pub(super) fn new(
        expected: &mut FeatureRegistry,
        policy: &FeaturePolicy,
        dimension: u32,
        device_index: usize,
        inputs: DeviceFeatureContextInputs<'_>,
        f08_max_n: Option<u32>,
        f08_max_changed_rows: Option<u32>,
        f08_max_component_events: Option<u32>,
        f08_max_history_counts: Option<u32>,
    ) -> Result<Self, String> {
        let selected = |family, ordinals: &[u8]| {
            policy
                .enabled_ids
                .iter()
                .any(|id| id.family() == family && ordinals.contains(&id.ordinal()))
        };
        let tokens = policy.schema_id == crate::schema::MISSING_TOKEN_LAYOUT_ID;
        let mut unsupported_at_width = std::collections::HashSet::new();
        let mut usable = |family: char, ordinals: &[u8], matches: bool| {
            if tokens && !matches {
                unsupported_at_width.extend(
                    policy
                        .enabled_ids
                        .iter()
                        .copied()
                        .filter(|id| id.family() == family && ordinals.contains(&id.ordinal())),
                );
                false
            } else {
                true
            }
        };
        let any_contextual = selected('C', &[2, 3, 4, 5, 6, 7, 8, 9, 10])
            || selected('D', &[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            || selected('E', &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            || selected('F', &[6, 9, 10])
            || selected('I', &[1, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
        if any_contextual && policy.symmetry_contract != "augmentation_only" {
            return Err("physical GPU context producers require augmentation_only symmetry".into());
        }
        let cuts = if selected('C', &[2, 3, 4]) {
            crate::register_labeled_cuts(expected, inputs.labeled_cuts.to_vec())
                .map_err(|e| e.0)?;
            usable(
                'C',
                &[2, 3, 4],
                inputs
                    .labeled_cuts
                    .iter()
                    .any(|c| c.dimension() == dimension),
            )
            .then(|| DeviceLabeledCutContext::new(dimension, inputs.labeled_cuts, device_index))
            .transpose()?
        } else {
            None
        };
        let partial_shear = selected('C', &[5]);
        let paid_basis = if selected('D', &[9, 10, 11, 12, 13, 14]) {
            let input = inputs
                .paid_basis
                .ok_or("D09-D14 require explicit paid-basis contexts and configuration")?;
            crate::register_paid_basis_features(expected, input.contexts.to_vec(), input.config)
                .map_err(|e| e.0)?;
            Some(crate::device_paid_basis::DevicePaidBasisPlan::new(
                dimension,
                device_index,
                input.contexts,
                input.config,
                policy.max_bytes,
            )?)
        } else {
            None
        };
        let basis_gradient = selected('E', &[1, 4, 5]);
        let zero_excess = if selected('F', &[7, 8]) {
            let input = inputs
                .guarded
                .ok_or("F07/F08 require explicit guarded recognizer limits")?;
            crate::register_guarded_zero_excess_with_budget(
                expected,
                input.max_n,
                input.max_entries,
            )
            .map_err(|e| e.0)?;
            Some(crate::device_zero_excess::DeviceZeroExcessPlan::new(
                dimension,
                device_index,
                input.max_n,
            )?)
        } else {
            None
        };
        let one_excess = if selected('F', &[8]) {
            let input = inputs
                .guarded
                .ok_or("F08 requires explicit guarded recognizer limits")?;
            let max_n = f08_max_n.unwrap_or(input.max_n);
            if max_n < 2 || max_n > input.max_n {
                return Err("F08 width limit must be between two and guarded max_n".into());
            }
            crate::guarded_features::register_guarded_one_excess_with_limits(
                expected, max_n, input.max_entries, f08_max_changed_rows, f08_max_component_events, f08_max_history_counts)
                .map_err(|e| e.0)?;
            Some(crate::device_one_excess::DeviceOneExcessPlan::new_with_limits(
                dimension,
                device_index,
                max_n,
                input.max_entries,
                f08_max_changed_rows,
                f08_max_component_events,
                f08_max_history_counts,
            )?)
        } else {
            None
        };
        let mut column_diffusion = None;
        let mut pair_diffusion = None;
        let mut shared_word = None;
        if selected('G', &[1, 2, 3, 4, 6, 7, 8, 9, 10]) {
            let config = inputs.diffusion.ok_or("missing diffusion configuration")?;
            crate::register_diffusion(expected, config).map_err(|e| e.0)?;
            if selected('G', &[7, 8, 9, 10]) {
                shared_word = Some(super::shared_word::SharedWord::new(
                    dimension,
                    device_index,
                    policy.seed,
                    config,
                )?);
            }
            if selected('G', &[1, 2, 3, 4]) {
                column_diffusion = Some(super::column_diffusion::ColumnDiffusion::new(
                    dimension,
                    device_index,
                    config,
                )?);
            }
            if selected('G', &[6]) {
                pair_diffusion = Some(super::pair_diffusion::PairDiffusion::new(
                    dimension,
                    device_index,
                    config,
                )?);
            }
        }
        let guarded_pair = if selected('F', &[2, 6]) {
            let input = inputs
                .guarded
                .ok_or("F02/F06 require explicit guarded recognizer limits")?;
            crate::register_guarded_pair_f02_view(expected, input.max_n, input.max_entries)
                .map_err(|e| e.0)?;
            crate::register_guarded_pair(expected, input.max_n, input.max_entries)
                .map_err(|e| e.0)?;
            Some(crate::device_guarded_pair::DeviceGuardedPairPlan::new(
                dimension,
                device_index,
                input.max_n,
            )?)
        } else {
            None
        };
        let template_response = if selected('E', &[2, 3, 6, 7, 8, 9]) {
            let config = inputs
                .template_response
                .ok_or("E02/E03/E06-E09 require an explicit finite-template configuration")?;
            crate::register_template_response(expected, config.clone()).map_err(|e| e.0)?;
            Some(
                crate::device_template_response::DeviceTemplateResponsePlan::new(
                    dimension,
                    device_index,
                    config,
                    policy.max_bytes,
                )?,
            )
        } else {
            None
        };
        if basis_gradient {
            crate::register_basis_gradient(expected).map_err(|e| e.0)?;
        }
        let general_compiler = if selected('F', &[9, 10]) {
            crate::register_general_compiler_bounds(expected).map_err(|e| e.0)?;
            Some(crate::device_compiler::DeviceGeneralCompilerPlan::new(
                dimension,
                device_index,
                crate::GeneralCompilerLimits::default(),
            )?)
        } else {
            None
        };
        let stage26 = if selected('F', &[3, 4, 5]) {
            let input = inputs
                .stage26
                .ok_or("Stage26 requires explicit width/checker-pair limits")?;
            crate::register_stage26_features_with_budget(
                expected,
                input.max_n,
                input.max_checker_pairs,
            )
            .map_err(|e| e.0)?;
            Some(super::stage26::Stage26Plan::new(
                dimension,
                device_index,
                input,
            )?)
        } else {
            None
        };
        if partial_shear {
            crate::register_partial_shear_rank(expected).map_err(|e| e.0)?;
        }
        let frequency = if selected('I', &[7]) {
            let input = inputs
                .frequency
                .ok_or("I07 requires explicit frequency contexts and limits")?;
            crate::register_frequency_contexts(
                expected,
                input.contexts.to_vec(),
                input.max_width,
                input.max_frequency_bits,
            )
            .map_err(|e| e.0)?;
            Some(super::frequency::FrequencyPlan::new(
                dimension,
                device_index,
                input,
            )?)
        } else {
            None
        };
        let intersections = if selected('C', &[6, 7]) {
            crate::register_restricted_intersections(
                expected,
                inputs.restricted_intersections.to_vec(),
            )
            .map_err(|e| e.0)?;
            usable(
                'C',
                &[6, 7],
                inputs
                    .restricted_intersections
                    .iter()
                    .any(|c| c.dimension() == dimension),
            )
            .then(|| {
                DeviceRestrictedIntersectionContext::new(
                    dimension,
                    inputs.restricted_intersections,
                    device_index,
                )
            })
            .transpose()?
        } else {
            None
        };
        let increments = if selected('C', &[9]) {
            crate::register_rank_increment_probes(expected, inputs.rank_increment_probes.to_vec())
                .map_err(|e| e.0)?;
            usable(
                'C',
                &[9],
                inputs
                    .rank_increment_probes
                    .iter()
                    .any(|c| c.dimension() == dimension),
            )
            .then(|| {
                DeviceRankIncrementContext::new(
                    dimension,
                    inputs.rank_increment_probes,
                    device_index,
                )
            })
            .transpose()?
        } else {
            None
        };
        let residual = if selected('C', &[10]) {
            let input = inputs
                .selected_residual
                .ok_or("C10 requires an explicit alphabet and selected actions")?;
            let extras = inputs.selected_residual_additional_contexts;
            if extras.is_empty() {
                crate::register_selected_residual_ranks(expected, input.alphabet, input.gates.to_vec())
                    .map_err(|e| e.0)?;
            } else {
                let mut profiles = Vec::with_capacity(extras.len() + 1);
                profiles.push(crate::SelectedResidualContext {
                    alphabet: input.alphabet.clone(),
                    gates: input.gates.to_vec(),
                });
                profiles.extend_from_slice(extras);
                crate::register_selected_residual_ranks_multi(expected, profiles).map_err(|e| e.0)?;
            }
            let matching = if input.alphabet.dimension() == dimension {
                Some(input)
            } else {
                extras.iter().find(|context| context.alphabet.dimension() == dimension)
                    .map(|context| SelectedResidualPlanInputs {
                        alphabet: &context.alphabet,
                        gates: &context.gates,
                    })
            };
            if !tokens && matching.is_none() {
                return Err("C10 alphabet has the wrong width".into());
            }
            usable('C', &[10], matching.is_some())
                .then(|| {
                    let input = matching.expect("width checked above");
                    DeviceSelectedResidualContext::new(input.alphabet, input.gates, device_index)
                })
                .transpose()?
        } else {
            None
        };
        let mut block_histogram = None;
        let endpoint_response = if selected('I', &[10]) {
            let config = inputs
                .endpoint_response
                .ok_or("I10 requires an explicit finite-horizon configuration")?;
            crate::register_tiny_i10(expected, config).map_err(|e| e.0)?;
            Some(
                crate::device_endpoint_response::DeviceEndpointResponsePlan::new(
                    dimension,
                    device_index,
                    config,
                )?,
            )
        } else {
            None
        };
        let quadratic = if selected('I', &[9]) {
            let input = inputs
                .quadratic
                .ok_or("I09 requires explicit quadratic contexts and limits")?;
            crate::register_i09_quadratic_response(expected, input.contexts.to_vec(), input.limits)
                .map_err(|e| e.0)?;
            Some(crate::device_quadratic::DeviceQuadraticPlan::new(
                dimension,
                device_index,
                input.contexts,
                input.limits,
            )?)
        } else {
            None
        };
        let positive_packet = if selected('I', &[8]) {
            let input = inputs
                .positive_packet
                .ok_or("I08 requires explicit positive-packet contexts and limits")?;
            crate::register_positive_packet_contexts(
                expected,
                input.contexts.to_vec(),
                input.max_n,
                input.max_slots,
                input.max_monomials,
                input.max_factor_entries,
            )
            .map_err(|e| e.0)?;
            Some(super::positive_packet::PositivePacketPlan::new(
                dimension,
                device_index,
                input,
                policy.max_bytes,
            )?)
        } else {
            None
        };
        let bilinear = if selected('I', &[5]) {
            let input = inputs
                .bilinear
                .ok_or("I05 requires explicit bilinear contexts and limits")?;
            crate::register_bilinear_contexts(
                expected,
                input.contexts.to_vec(),
                input.max_dimension,
                input.max_matrix_bits,
            )
            .map_err(|e| e.0)?;
            Some(super::bilinear::BilinearPlan::new(
                dimension,
                device_index,
                input,
                policy.max_bytes,
            )?)
        } else {
            None
        };
        let shared_relation = if selected('I', &[6]) {
            let input = inputs
                .shared_relation
                .ok_or("I06 requires explicit shared-relation contexts and limits")?;
            crate::register_shared_relation_contexts(
                expected,
                input.contexts.to_vec(),
                input.max_vectors,
                input.max_coordinates,
                input.max_ports,
            )
            .map_err(|e| e.0)?;
            Some(super::shared_relation::SharedRelationPlan::new(
                dimension,
                device_index,
                input,
                policy.max_bytes,
            )?)
        } else {
            None
        };
        let blocks = if selected('D', &[1, 2, 4, 5, 6, 7, 8]) {
            let input = inputs
                .block_parities
                .ok_or("block-parity features require explicit physical blocks and limits")?;
            crate::register_block_parities(
                expected,
                input.contexts.to_vec(),
                input.max_patterns,
                input.max_pairs,
            )
            .map_err(|e| e.0)?;
            let count = input
                .contexts
                .iter()
                .filter(|c| c.dimension() == dimension)
                .count();
            usable('D', &[5], count == 1);
            if usable('D', &[1, 2, 4, 6, 7, 8], count > 0) {
                let context = DeviceBlockParityContext::new(
                    dimension,
                    input.contexts,
                    device_index,
                    input.max_patterns,
                )?;
                if !tokens && selected('D', &[5]) && context.context_count != 1 {
                    return Err(
                        "D05 requires exactly one physical block context at this width".into(),
                    );
                }
                if selected('D', &[1, 2, 4, 6, 7]) {
                    block_histogram = Some(BlockHistogramPlan::new(dimension, input, policy)?);
                }
                Some(context)
            } else {
                None
            }
        } else {
            None
        };
        let mut boundary_limited = false;
        let mut boundary_relation = None;
        let boundary = if selected('C', &[8]) || selected('I', &[1, 2, 3, 4, 11]) {
            let input = inputs
                .matrix_boundaries
                .ok_or("matrix-boundary features require explicit boundaries and limits")?;
            crate::register_matrix_boundaries(
                expected,
                input.boundaries.to_vec(),
                input.max_ports,
                input.max_equations,
            )
            .map_err(|e| e.0)?;
            let matching = input
                .boundaries
                .iter()
                .find(|spec| spec.dimension() == dimension);
            usable('I', &[1, 2, 3, 4, 11], matching.is_some());
            if usable('C', &[8], matching.is_some()) {
                let spec = matching.ok_or("no explicit matrix boundary matches this width")?;
                boundary_limited = spec.inputs().len().saturating_add(spec.outputs().len())
                    > input.max_ports
                    || spec
                        .outputs()
                        .len()
                        .saturating_add(spec.zero_outputs().len())
                        > input.max_equations;
                if selected('I', &[1, 2, 3, 4, 11]) {
                    boundary_relation = Some(BoundaryPlan::new(spec, device_index)?);
                }
                Some(DeviceMatrixBoundaryContext::new(spec, device_index)?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            unsupported_at_width,
            dimension,
            cuts,
            partial_shear,
            paid_basis,
            basis_gradient,
            template_response,
            general_compiler,
            stage26,
            guarded_pair,
            zero_excess,
            one_excess,
            column_diffusion,
            pair_diffusion,
            shared_word,
            frequency,
            shared_relation,
            bilinear,
            positive_packet,
            quadratic,
            endpoint_response,
            intersections,
            increments,
            residual,
            blocks,
            block_histogram,
            boundary,
            boundary_limited,
            boundary_relation,
        })
    }
}
