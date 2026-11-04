mod entry;
mod validation;

use entry::FeatureEntryContext;

use super::registry::FeatureRegistry;
use super::{validate_policy, FeatureBuildError, FeatureComputation, FeaturePolicy};
use crate::packet::FeaturePacket;
use crate::schema::{feature_spec, FeatureId, FEATURE_COUNT};
use crate::state::MatrixState;
use crate::structural;

impl FeatureRegistry {
    pub fn build(
        &self,
        states: &[MatrixState],
        policy: &FeaturePolicy,
    ) -> Result<FeaturePacket, FeatureBuildError> {
        validate_policy(policy)?;
        if states.is_empty() {
            return Err(FeatureBuildError(
                "cannot build an empty feature batch".into(),
            ));
        }
        for (index, state) in states.iter().enumerate() {
            if state.alphabet_id().trim().is_empty() {
                return Err(FeatureBuildError(format!(
                    "state {index} has no alphabet identity"
                )));
            }
        }
        for id in &policy.enabled_ids {
            let spec = self
                .specs
                .get(id)
                .ok_or_else(|| FeatureBuildError(format!("unknown feature ID {id}")))?;
            if spec.requires_solver_assisted && !policy.solver_assisted {
                return Err(FeatureBuildError(format!(
                    "{id} requires a solver-assisted feature policy"
                )));
            }
            if policy.symmetry_contract == "exact_relabel" {
                if let Some(provider) = self.providers.get(id) {
                    if provider.symmetry_contract() != "exact_relabel" {
                        return Err(FeatureBuildError(format!(
                            "{id} provider symmetry contract mismatch"
                        )));
                    }
                }
            }
        }

        let nodes = states
            .iter()
            .map(|state| state.dimension() as usize)
            .max()
            .unwrap_or(0);
        let valid_nodes = states
            .iter()
            .flat_map(|state| {
                (0..nodes).map(move |node| u8::from(node < state.dimension() as usize))
            })
            .collect::<Vec<_>>();
        let estimated_bytes = states
            .iter()
            .try_fold(0usize, |total, state| {
                let n = state.dimension() as usize;
                n.checked_mul(n)
                    .and_then(|square| square.checked_mul(1024))
                    .and_then(|matrix| {
                        n.checked_mul(256)
                            .and_then(|vectors| matrix.checked_add(vectors))
                    })
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| FeatureBuildError("feature workspace estimate overflow".into()))?;
        let budget_exhausted_before_compute = policy
            .max_bytes
            .is_some_and(|limit| estimated_bytes > limit);

        // The source implementation computes its structural bundle once per
        // state; selected IDs are read from the shared result.
        let needs_structural = !budget_exhausted_before_compute
            && policy
                .enabled_ids
                .iter()
                .any(|id| structural::is_builtin(*id));
        let structural_rows = if needs_structural {
            states
                .iter()
                .map(|state| {
                    structural::compute(state, &policy.enabled_ids)
                        .map(Some)
                        .map_err(FeatureBuildError)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; states.len()]
        };

        let enabled = policy
            .enabled_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut remaining_evaluations = policy.evaluations;
        let mut entries = Vec::with_capacity(FEATURE_COUNT);
        for index in 0..FEATURE_COUNT {
            let id = FeatureId::from_index(index)
                .ok_or_else(|| FeatureBuildError("feature manifest index invalid".into()))?;
            let spec = self
                .specs
                .get(&id)
                .cloned()
                .unwrap_or_else(|| feature_spec(id));
            entries.push(self.build_entry(
                spec,
                FeatureEntryContext {
                    states,
                    nodes,
                    enabled: enabled.contains(&id),
                    budget_exhausted: budget_exhausted_before_compute,
                    remaining_evaluations: &mut remaining_evaluations,
                    policy,
                    structural_rows: &structural_rows,
                },
            )?);
        }
        let policy_id = policy.id()?;
        FeaturePacket::new(
            entries,
            states,
            nodes,
            valid_nodes,
            policy.schema_id.clone(),
            policy_id,
        )
        .map_err(FeatureBuildError::from)
    }
}
