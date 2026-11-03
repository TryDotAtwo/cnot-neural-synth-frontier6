use super::*;
use crate::alphabet::CnotAlphabet;
use crate::features::FeatureProviderContexts;
use crate::matrix_boundary::{MatrixBoundarySpec, RationalWeight};
use crate::{
    BilinearContext, BlockContext, CnotGate, FrequencyContext, I09Context, I09Limits, I10Config,
    LabeledCut, LocalBooleanFactor, OptionalGatePacket, PaidBasisBlockContext, PaidShearConfig,
    RankIncrementProbe, RationalProbability, RestrictedIntersection, SharedRelationContext,
    TemplateChoice, TemplateChoiceKind, TemplateResponseBudgets, TemplateResponseConfig,
};
use num_bigint::BigUint;

impl GateSpec {
    pub(crate) fn build(self) -> CnotGate {
        CnotGate {
            control: self.control,
            target: self.target,
        }
    }
}

impl AlphabetSpec {
    pub(crate) fn build(&self) -> Result<CnotAlphabet, String> {
        CnotAlphabet::new(
            self.dimension,
            self.id.clone(),
            self.gates.iter().copied().map(GateSpec::build).collect(),
            self.costs.clone(),
        )
        .map_err(|e| e.to_string())
    }
}

impl FeatureContextsSpec {
    pub(crate) fn build(self) -> Result<FeatureProviderContexts, String> {
        let alphabet = self.alphabet.build()?;
        let labeled_cuts = self
            .labeled_cuts
            .into_iter()
            .map(|x| LabeledCut::new(x.n, x.a).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let restricted_intersections = self
            .restricted_intersections
            .into_iter()
            .map(|x| {
                RestrictedIntersection::new(x.n, x.left, x.right, x.coordinates)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rank_increment_probes = self
            .rank_increment_probes
            .into_iter()
            .map(|x| {
                RankIncrementProbe::new(x.n, x.span_rows, x.coordinates, x.candidate_support)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let block_parity_contexts = self
            .block_parity_contexts
            .into_iter()
            .map(|x| BlockContext::new(x.n, x.sources, x.targets).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let matrix_boundaries = self
            .matrix_boundaries
            .into_iter()
            .map(MatrixBoundarySpecInput::build)
            .collect::<Result<Vec<_>, _>>()?;
        let paid_basis_contexts = self
            .paid_basis_contexts
            .into_iter()
            .map(|x| {
                PaidBasisBlockContext::new(x.dimension, x.source_wires, x.target_wires)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let template_response = self.template_response.build()?;
        let bilinear_contexts = self
            .bilinear_contexts
            .into_iter()
            .map(|x| {
                BilinearContext::new(
                    x.n,
                    x.rows,
                    x.columns,
                    x.hidden_width,
                    x.u_rows,
                    x.v_rows,
                    x.q,
                    x.a0_rows,
                    x.d0_rows,
                )
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_relation_contexts = self
            .shared_relation_contexts
            .into_iter()
            .map(|x| {
                SharedRelationContext::new(x.n, x.rows, x.coordinates, x.covectors)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let frequency_contexts = self
            .frequency_contexts
            .into_iter()
            .map(|x| {
                FrequencyContext::new(x.n, x.sources, x.internal, x.targets, x.central_rows)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let positive_packet_contexts = self
            .positive_packet_contexts
            .into_iter()
            .map(OptionalGatePacketSpec::build)
            .collect::<Result<Vec<_>, _>>()?;
        let i09_contexts = self
            .i09_contexts
            .into_iter()
            .map(|x| {
                I09Context::new(
                    x.n,
                    x.horizon,
                    x.cut,
                    x.prefix_bits,
                    x.allowed_gates.into_iter().map(|[c, t]| (c, t)).collect(),
                )
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FeatureProviderContexts {
            max_n: self.max_n,
            labeled_cuts,
            restricted_intersections,
            rank_increment_probes,
            alphabet,
            selected_residual_gates: self
                .selected_residual_gates
                .into_iter()
                .map(GateSpec::build)
                .collect(),
            block_parity_contexts,
            max_parity_patterns: self.max_parity_patterns,
            max_parity_pairs: self.max_parity_pairs,
            matrix_boundaries,
            max_boundary_ports: self.max_boundary_ports,
            max_boundary_equations: self.max_boundary_equations,
            paid_basis_contexts,
            paid_shear_config: PaidShearConfig {
                local_width: self.paid_shear_config.local_width,
                sweeps: self.paid_shear_config.sweeps,
            },
            template_response,
            bilinear_contexts,
            shared_relation_contexts,
            frequency_contexts,
            positive_packet_contexts,
            i09_contexts,
            i09_limits: I09Limits {
                max_variables: self.i09_limits.max_variables,
                max_equations: self.i09_limits.max_equations,
                max_bytes: self.i09_limits.max_bytes,
            },
            i10_config: I10Config {
                max_n: self.i10_config.max_n,
                max_degree: self.i10_config.max_degree,
                max_layer_states: self.i10_config.max_layer_states,
                max_transitions: self.i10_config.max_transitions,
            },
        })
    }
}

impl MatrixBoundarySpecInput {
    pub(crate) fn build(self) -> Result<MatrixBoundarySpec, String> {
        let mut spec = MatrixBoundarySpec::new(self.n, self.inputs, self.outputs)
            .map_err(|e| e.to_string())?
            .with_protected_inputs(self.protected_inputs)
            .map_err(|e| e.to_string())?
            .with_zero_outputs(self.zero_outputs)
            .map_err(|e| e.to_string())?;
        if !self.weights.is_empty() {
            let weights = self
                .weights
                .into_iter()
                .map(|[first, second]| {
                    Ok((
                        RationalWeight::new(first.numerator, first.denominator)
                            .map_err(|e| e.to_string())?,
                        RationalWeight::new(second.numerator, second.denominator)
                            .map_err(|e| e.to_string())?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            spec = spec.with_weights(weights).map_err(|e| e.to_string())?;
        }
        Ok(spec)
    }
}

impl TemplateResponseSpec {
    pub(crate) fn build(self) -> Result<TemplateResponseConfig, String> {
        let choices = self
            .choices
            .into_iter()
            .map(|x| TemplateChoice::new(x.kind.into(), x.control, x.target))
            .collect();
        let points = self
            .points
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|x| {
                        RationalProbability::new(x.numerator, x.denominator)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        TemplateResponseConfig::with_budgets(
            choices,
            points,
            TemplateResponseBudgets {
                max_width: self.budgets.max_width,
                max_assignments: self.budgets.max_assignments,
                max_evaluations: self.budgets.max_evaluations,
                max_bytes: self.budgets.max_bytes,
            },
        )
        .map_err(|e| e.to_string())
    }
}

impl From<ChoiceKindSpec> for TemplateChoiceKind {
    fn from(x: ChoiceKindSpec) -> Self {
        match x {
            ChoiceKindSpec::Add => Self::Add,
            ChoiceKindSpec::Swap => Self::Swap,
        }
    }
}

impl OptionalGatePacketSpec {
    pub(crate) fn build(self) -> Result<OptionalGatePacket, String> {
        let gates = self.gates.into_iter().map(|[c, t]| (c, t)).collect();
        let factors = self
            .factors
            .into_iter()
            .map(|x| {
                let allowed = x
                    .allowed
                    .iter()
                    .map(|s| {
                        s.parse::<BigUint>()
                            .map_err(|e| format!("invalid decimal BigUint: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                LocalBooleanFactor::new(x.scope, allowed).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        OptionalGatePacket::new(self.n, gates, factors).map_err(|e| e.to_string())
    }
}
