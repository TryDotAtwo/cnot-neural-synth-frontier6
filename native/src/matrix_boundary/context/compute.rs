mod encoding;
mod relations;
mod size;

use super::MatrixBoundaryContext;
use crate::features::FeatureComputation;
use crate::matrix_boundary::MATRIX_BOUNDARY_IDS;
use crate::packet::FeatureStatus;
use crate::schema::FeatureId;
use crate::state::MatrixState;
use std::collections::HashMap;

impl MatrixBoundaryContext {
    pub(super) fn compute(
        &self,
        state: &MatrixState,
        max_bytes: Option<usize>,
    ) -> HashMap<FeatureId, FeatureComputation> {
        let Some(spec) = self.specs.iter().find(|spec| spec.n == state.dimension()) else {
            return unavailable_all(
                FeatureStatus::Unsupported,
                "no_explicit_boundary_for_matrix_width",
            );
        };
        let ports = spec.inputs.len().saturating_add(spec.outputs.len());
        let equations = spec.outputs.len().saturating_add(spec.zero_outputs.len());
        if ports > self.max_ports || equations > self.max_equations {
            return unavailable_all(
                FeatureStatus::ResourceLimit,
                "matrix_boundary_port_or_equation_budget_exceeded",
            );
        }

        let relation_summary = match relations::build(spec, state, ports, equations) {
            Ok(summary) => summary,
            Err(reason) => return unavailable_all(FeatureStatus::Unsupported, reason),
        };
        let visible_rank = relation_summary.visible_rank;
        let relation_rows = relation_summary.relation_rows;
        let relation_rank = relation_summary.relation_rank;
        let protected_rank = relation_summary.protected_rank;
        let Some(byte_count) =
            size::estimated_output_bytes(ports, state.dimension() as usize, relation_rank)
        else {
            return unavailable_all(
                FeatureStatus::ResourceLimit,
                "matrix_boundary_output_size_overflow",
            );
        };
        if max_bytes.is_some_and(|budget| byte_count > budget) {
            return unavailable_all(
                FeatureStatus::ResourceLimit,
                "matrix_boundary_output_exceeds_policy_byte_budget",
            );
        }

        encoding::encode(
            spec,
            state.dimension() as usize,
            ports,
            &relation_rows,
            visible_rank,
            relation_rank,
            protected_rank,
        )
    }
}

fn unavailable_all(status: FeatureStatus, reason: &str) -> HashMap<FeatureId, FeatureComputation> {
    MATRIX_BOUNDARY_IDS
        .iter()
        .map(|(family, ordinal)| {
            (
                FeatureId::new(*family, *ordinal).expect("matrix-boundary ID is canonical"),
                FeatureComputation::Unavailable {
                    status,
                    reason: reason.into(),
                },
            )
        })
        .collect()
}
