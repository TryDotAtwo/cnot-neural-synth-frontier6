//! Registration and execution of contextual exact-rank feature providers.

use super::contexts::{LabeledCut, RankIncrementProbe, RestrictedIntersection};
use super::rank_math::{projected_rank, rank_binary_block};
use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::shear::partial_shear;
use crate::state::{CnotGate, MatrixState};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Clone, Copy)]
enum RankChannel {
    AToB,
    BToA,
    ResidualAA,
}

struct LabeledCutRankProducer {
    spec: FeatureSpec,
    cuts: Vec<LabeledCut>,
    channel: RankChannel,
}

struct PartialShearRankProducer {
    spec: FeatureSpec,
}

impl FeatureProducer for PartialShearRankProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "partial_shear_rank_requires_augmentation_only_symmetry",
            );
        }
        let decomposition = match partial_shear(state) {
            Ok(Some(decomposition)) => decomposition,
            Ok(None) => {
                return unavailable(
                    FeatureStatus::Inapplicable,
                    "no_nonzero_partial_shear_block",
                )
            }
            Err(_) => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    "partial_shear_decomposition_failed",
                )
            }
        };
        let rank = match rank_binary_block(&decomposition.block) {
            Ok(rank) => rank,
            Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
        };
        if rank > (1 << 24) {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "partial_shear_rank_exceeds_exact_f32_integer_range",
            );
        }
        FeatureComputation::Computed(FeatureValue {
            shape: vec![1],
            values: vec![rank as f32],
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

#[derive(Clone)]
enum ContextualRankContext {
    RestrictedIntersection {
        probes: Vec<RestrictedIntersection>,
        transpose: bool,
    },
    RankIncrement(Vec<RankIncrementProbe>),
    SelectedResidual {
        alphabet: CnotAlphabet,
        gates: Vec<CnotGate>,
    },
}

struct ContextualRankProducer {
    spec: FeatureSpec,
    context: ContextualRankContext,
}

impl FeatureProducer for ContextualRankProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "restricted_rank_probes_require_augmentation_only_symmetry",
            );
        }
        match &self.context {
            ContextualRankContext::RestrictedIntersection { probes, transpose } => {
                let relevant = probes
                    .iter()
                    .filter(|probe| probe.n == state.dimension())
                    .collect::<Vec<_>>();
                if relevant.is_empty() {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "no_restricted_intersection_probe_for_matrix_width",
                    );
                }
                let Some((mut values, mut incidence)) = group_buffers(state, relevant.len()) else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "rank-probe output too large",
                    );
                };
                for probe in &relevant {
                    let left = match projected_rank(
                        state,
                        &probe.left,
                        &probe.coordinates,
                        false,
                        *transpose,
                        None,
                    ) {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    let right = match projected_rank(
                        state,
                        &probe.right,
                        &probe.coordinates,
                        false,
                        *transpose,
                        None,
                    ) {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    let Some(joint_len) = probe.left.len().checked_add(probe.right.len()) else {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "joint rank row count overflow",
                        );
                    };
                    let mut joint_rows = Vec::new();
                    if joint_rows.try_reserve_exact(joint_len).is_err() {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "joint rank row allocation failed",
                        );
                    }
                    joint_rows.extend_from_slice(&probe.left);
                    joint_rows.extend_from_slice(&probe.right);
                    let joint = match projected_rank(
                        state,
                        &joint_rows,
                        &probe.coordinates,
                        false,
                        *transpose,
                        None,
                    ) {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    let Some(intersection) = left
                        .checked_add(right)
                        .and_then(|sum| sum.checked_sub(joint))
                    else {
                        return unavailable(
                            FeatureStatus::Unsupported,
                            "invalid subspace rank identity",
                        );
                    };
                    if intersection > (1 << 24) {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "intersection_dimension_exceeds_exact_f32_integer_range",
                        );
                    }
                    values.push(intersection as f32);
                    append_incidence(
                        state.dimension(),
                        [
                            probe.left.as_slice(),
                            probe.right.as_slice(),
                            probe.coordinates.as_slice(),
                        ],
                        &mut incidence,
                    );
                }
                computed_group(relevant.len(), values, incidence)
            }
            ContextualRankContext::RankIncrement(probes) => {
                let relevant = probes
                    .iter()
                    .filter(|probe| probe.n == state.dimension())
                    .collect::<Vec<_>>();
                if relevant.is_empty() {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "no_rank-increment_probe_for_matrix_width",
                    );
                }
                let Some((mut values, mut incidence)) = group_buffers(state, relevant.len()) else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "rank-probe output too large",
                    );
                };
                for probe in &relevant {
                    let baseline = match projected_rank(
                        state,
                        &probe.span_rows,
                        &probe.coordinates,
                        false,
                        false,
                        None,
                    ) {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    let augmented = match projected_rank(
                        state,
                        &probe.span_rows,
                        &probe.coordinates,
                        false,
                        false,
                        Some(&probe.candidate_support),
                    ) {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    let Some(increment) = augmented.checked_sub(baseline) else {
                        return unavailable(
                            FeatureStatus::Unsupported,
                            "candidate reduced restricted rank",
                        );
                    };
                    if increment > (1 << 24) {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "rank_increment_exceeds_exact_f32_integer_range",
                        );
                    }
                    values.push(increment as f32);
                    append_incidence(
                        state.dimension(),
                        [
                            probe.span_rows.as_slice(),
                            probe.coordinates.as_slice(),
                            probe.candidate_support.as_slice(),
                        ],
                        &mut incidence,
                    );
                }
                computed_group(relevant.len(), values, incidence)
            }
            ContextualRankContext::SelectedResidual { alphabet, gates } => {
                if state.dimension() != alphabet.dimension() || state.alphabet_id() != alphabet.id()
                {
                    return unavailable(
                        FeatureStatus::Unsupported,
                        "selected_residual_alphabet_differs_from_matrix_state",
                    );
                }
                let Some((mut values, mut incidence)) =
                    group_buffers_with_roles(state, gates.len(), 2)
                else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "rank-probe output too large",
                    );
                };
                let Some(wires) = all_wires(state.dimension()) else {
                    return unavailable(
                        FeatureStatus::ResourceLimit,
                        "wire-index allocation failed",
                    );
                };
                for gate in gates {
                    if alphabet.cost(*gate).is_none() {
                        return unavailable(
                            FeatureStatus::Unsupported,
                            "selected_gate_missing_from_physical_alphabet",
                        );
                    }
                    let mut transformed = state.clone();
                    if transformed.apply_cnot(*gate).is_err() {
                        return unavailable(
                            FeatureStatus::Unsupported,
                            "selected_gate_invalid_for_matrix_width",
                        );
                    }
                    let rank = match projected_rank(&transformed, &wires, &wires, true, false, None)
                    {
                        Ok(rank) => rank,
                        Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
                    };
                    if rank > (1 << 24) {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "selected_residual_rank_exceeds_exact_f32_integer_range",
                        );
                    }
                    values.push(rank as f32);
                    for wire in 0..state.dimension() {
                        incidence.push(f32::from(gate.control == wire));
                        incidence.push(f32::from(gate.target == wire));
                    }
                }
                computed_group(gates.len(), values, incidence)
            }
        }
    }
}

impl FeatureProducer for LabeledCutRankProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return unavailable(
                FeatureStatus::Unsupported,
                "labeled_cuts_require_augmentation_only_symmetry",
            );
        }

        let relevant = self
            .cuts
            .iter()
            .filter(|cut| cut.dimension() == state.dimension())
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            return unavailable(
                FeatureStatus::Unsupported,
                "no_labeled_cut_for_matrix_width",
            );
        }

        let Some(incidence_len) = relevant
            .len()
            .checked_mul(state.dimension() as usize)
            .and_then(|size| size.checked_mul(2))
        else {
            return unavailable(FeatureStatus::ResourceLimit, "cut incidence size overflow");
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(relevant.len()).is_err() {
            return unavailable(FeatureStatus::ResourceLimit, "cut value allocation failed");
        }
        let mut incidence = Vec::new();
        if incidence.try_reserve_exact(incidence_len).is_err() {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "cut incidence allocation failed",
            );
        }
        for cut in &relevant {
            let side_b = cut.side_b();
            let (rows, columns, subtract_identity) = match self.channel {
                RankChannel::AToB => (cut.side_a(), side_b.as_slice(), false),
                RankChannel::BToA => (side_b.as_slice(), cut.side_a(), false),
                RankChannel::ResidualAA => (cut.side_a(), cut.side_a(), true),
            };
            let rank = match projected_rank(state, rows, columns, subtract_identity, false, None) {
                Ok(rank) => rank,
                Err(reason) => return unavailable(FeatureStatus::ResourceLimit, reason),
            };
            if rank > (1 << 24) {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "rank_exceeds_exact_f32_integer_range",
                );
            }
            values.push(rank as f32);
            for wire in 0..state.dimension() {
                incidence.push(f32::from(cut.side_a().binary_search(&wire).is_ok()));
                incidence.push(f32::from(side_b.binary_search(&wire).is_ok()));
            }
        }

        FeatureComputation::Computed(FeatureValue {
            shape: vec![relevant.len(), 1],
            values,
            token_valid: vec![1; relevant.len()],
            incidence,
        })
    }
}

fn group_buffers(state: &MatrixState, groups: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    group_buffers_with_roles(state, groups, 3)
}

fn group_buffers_with_roles(
    state: &MatrixState,
    groups: usize,
    role_count: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let incidence_len = groups
        .checked_mul(state.dimension() as usize)?
        .checked_mul(role_count)?;
    let mut values = Vec::new();
    values.try_reserve_exact(groups).ok()?;
    let mut incidence = Vec::new();
    incidence.try_reserve_exact(incidence_len).ok()?;
    Some((values, incidence))
}

fn append_incidence(n: u32, roles: [&[u32]; 3], output: &mut Vec<f32>) {
    for wire in 0..n {
        for role in roles {
            output.push(f32::from(role.binary_search(&wire).is_ok()));
        }
    }
}

fn all_wires(n: u32) -> Option<Vec<u32>> {
    let length = usize::try_from(n).ok()?;
    let mut wires = Vec::new();
    wires.try_reserve_exact(length).ok()?;
    wires.extend(0..n);
    Some(wires)
}

fn computed_group(groups: usize, values: Vec<f32>, incidence: Vec<f32>) -> FeatureComputation {
    FeatureComputation::Computed(FeatureValue {
        shape: vec![groups, 1],
        values,
        token_valid: vec![1; groups],
        incidence,
    })
}

/// Registers C02, C03, and C04 for an explicit ordered set of labeled cuts.
/// Repeated cuts are rejected because their group-token positions would be
/// ambiguous. The provider version is bound to a SHA-256 of the exact context.
pub fn register_labeled_cuts(
    registry: &mut FeatureRegistry,
    cuts: Vec<LabeledCut>,
) -> Result<(), FeatureBuildError> {
    if cuts.is_empty() {
        return Err(FeatureBuildError(
            "at least one labeled cut is required".into(),
        ));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(cuts.len())
        .map_err(|_| FeatureBuildError("labeled-cut validation allocation failed".into()))?;
    if cuts.iter().any(|cut| !seen.insert(cut)) {
        return Err(FeatureBuildError("labeled cuts must be unique".into()));
    }

    let current_specs = registry.specs();
    for ordinal in 2..=4 {
        let id = FeatureId::new(b'C', ordinal)
            .expect("C02 through C04 are in the canonical feature manifest");
        if current_specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }

    let mut digest = Sha256::new();
    digest.update(b"cnot103/labeled-cut-ranks/v1\0");
    digest.update((cuts.len() as u64).to_le_bytes());
    for cut in &cuts {
        digest.update(cut.dimension().to_le_bytes());
        digest.update((cut.side_a().len() as u64).to_le_bytes());
        for wire in cut.side_a() {
            digest.update(wire.to_le_bytes());
        }
    }
    let context = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let version = format!("labeled_cut_ranks_v1:{context}");

    for (ordinal, channel) in [
        RankChannel::AToB,
        RankChannel::BToA,
        RankChannel::ResidualAA,
    ]
    .into_iter()
    .enumerate()
    {
        let id = FeatureId::new(b'C', ordinal as u8 + 2)
            .expect("C02 through C04 are in the canonical feature manifest");
        let mut spec = feature_spec(id);
        spec.scope = "explicit_labeled_cut;augmentation_only;rank_not_distance".into();
        spec.producer_version = version.clone();
        spec.dependencies = vec!["M".into(), "labeled_cut_partition".into()];
        spec.exactness = "exact".into();
        spec.cost_tier = "T1".into();
        spec.wire_roles = vec!["cut_A".into(), "cut_B".into()];
        spec.requires_solver_assisted = false;
        debug_assert_eq!(spec.role, FeatureRole::Group);
        debug_assert_eq!(spec.channels, 1);
        registry.register(Box::new(LabeledCutRankProducer {
            spec,
            cuts: cuts.clone(),
            channel,
        }))?;
    }
    Ok(())
}

/// Registers C06/C07 for explicit coordinate-restricted row/column subspaces.
pub fn register_restricted_intersections(
    registry: &mut FeatureRegistry,
    probes: Vec<RestrictedIntersection>,
) -> Result<(), FeatureBuildError> {
    validate_unique_context(&probes, "restricted-intersection probes")?;
    let ids = [
        FeatureId::new(b'C', 6).unwrap(),
        FeatureId::new(b'C', 7).unwrap(),
    ];
    ensure_unconfigured(registry, &ids)?;
    let mut digest = Sha256::new();
    digest.update(b"cnot103/restricted-intersections/v1\0");
    digest.update((probes.len() as u64).to_le_bytes());
    for probe in &probes {
        digest.update(probe.n.to_le_bytes());
        hash_indices(&mut digest, &probe.left);
        hash_indices(&mut digest, &probe.right);
        hash_indices(&mut digest, &probe.coordinates);
    }
    let version = format!("restricted_intersections_v1:{}", finish_digest(digest));
    for (id, transpose) in ids.into_iter().zip([false, true]) {
        let spec = contextual_spec(
            id,
            version.clone(),
            "explicit_restricted_subspaces;augmentation_only",
            vec!["M".into(), "restricted_index_context".into()],
            vec![
                "left_span".into(),
                "right_span".into(),
                "coordinate_slice".into(),
            ],
        );
        registry.register(Box::new(ContextualRankProducer {
            spec,
            context: ContextualRankContext::RestrictedIntersection {
                probes: probes.clone(),
                transpose,
            },
        }))?;
    }
    Ok(())
}

/// Registers C09 for explicit restricted spans and candidate parity vectors.
pub fn register_rank_increment_probes(
    registry: &mut FeatureRegistry,
    probes: Vec<RankIncrementProbe>,
) -> Result<(), FeatureBuildError> {
    validate_unique_context(&probes, "rank-increment probes")?;
    let id = FeatureId::new(b'C', 9).unwrap();
    ensure_unconfigured(registry, &[id])?;
    let mut digest = Sha256::new();
    digest.update(b"cnot103/restricted-rank-increments/v1\0");
    digest.update((probes.len() as u64).to_le_bytes());
    for probe in &probes {
        digest.update(probe.n.to_le_bytes());
        hash_indices(&mut digest, &probe.span_rows);
        hash_indices(&mut digest, &probe.coordinates);
        hash_indices(&mut digest, &probe.candidate_support);
    }
    let version = format!("restricted_rank_increment_v1:{}", finish_digest(digest));
    let spec = contextual_spec(
        id,
        version,
        "explicit_restricted_span_and_candidate;augmentation_only",
        vec![
            "M".into(),
            "restricted_index_context".into(),
            "candidate_parity".into(),
        ],
        vec![
            "span_rows".into(),
            "coordinate_slice".into(),
            "candidate_support".into(),
        ],
    );
    registry.register(Box::new(ContextualRankProducer {
        spec,
        context: ContextualRankContext::RankIncrement(probes),
    }))
}

/// Registers C10 for explicit physical CNOT actions in the supplied alphabet.
pub fn register_selected_residual_ranks(
    registry: &mut FeatureRegistry,
    alphabet: &CnotAlphabet,
    gates: Vec<CnotGate>,
) -> Result<(), FeatureBuildError> {
    if gates.is_empty() {
        return Err(FeatureBuildError(
            "at least one selected physical CNOT is required".into(),
        ));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(gates.len())
        .map_err(|_| FeatureBuildError("selected-gate validation allocation failed".into()))?;
    for gate in &gates {
        if alphabet.cost(*gate).is_none() {
            return Err(FeatureBuildError(
                "selected CNOT must belong to the supplied physical alphabet".into(),
            ));
        }
        if !seen.insert(*gate) {
            return Err(FeatureBuildError(
                "selected CNOT gates must be unique".into(),
            ));
        }
    }

    let id = FeatureId::new(b'C', 10).unwrap();
    ensure_unconfigured(registry, &[id])?;
    let mut digest = Sha256::new();
    digest.update(b"cnot103/selected-residual-ranks/v1\0");
    digest.update(alphabet.dimension().to_le_bytes());
    digest.update((alphabet.id().len() as u64).to_le_bytes());
    digest.update(alphabet.id().as_bytes());
    digest.update((alphabet.gates().len() as u64).to_le_bytes());
    for gate in alphabet.gates() {
        digest.update(gate.control.to_le_bytes());
        digest.update(gate.target.to_le_bytes());
        digest.update(
            alphabet
                .cost(*gate)
                .expect("alphabet gates always have costs")
                .to_le_bytes(),
        );
    }
    digest.update((gates.len() as u64).to_le_bytes());
    for gate in &gates {
        digest.update(gate.control.to_le_bytes());
        digest.update(gate.target.to_le_bytes());
    }
    let version = format!("selected_residual_rank_v1:{}", finish_digest(digest));
    let spec = contextual_spec(
        id,
        version,
        "explicit_selected_physical_CNOT;rank_residual_not_distance",
        vec!["M".into(), "alphabet".into(), "selected_actions".into()],
        vec!["control".into(), "target".into()],
    );
    registry.register(Box::new(ContextualRankProducer {
        spec,
        context: ContextualRankContext::SelectedResidual {
            alphabet: alphabet.clone(),
            gates,
        },
    }))
}

/// Registers C05 for the source-defined deterministic partial-shear block.
pub fn register_partial_shear_rank(
    registry: &mut FeatureRegistry,
) -> Result<(), FeatureBuildError> {
    let id = FeatureId::new(b'C', 5).unwrap();
    ensure_unconfigured(registry, &[id])?;
    let mut spec = feature_spec(id);
    spec.scope = "selected_partial_elimination;augmentation_only".into();
    spec.producer_version = "selected_partial_shear_rank_v1".into();
    spec.dependencies = vec!["M".into(), "deterministic_partial_shear".into()];
    spec.exactness = "exact".into();
    spec.cost_tier = "T1".into();
    spec.wire_roles.clear();
    registry.register(Box::new(PartialShearRankProducer { spec }))
}

fn contextual_spec(
    id: FeatureId,
    producer_version: String,
    scope: &str,
    dependencies: Vec<String>,
    wire_roles: Vec<String>,
) -> FeatureSpec {
    let mut spec = feature_spec(id);
    spec.scope = scope.into();
    spec.producer_version = producer_version;
    spec.dependencies = dependencies;
    spec.exactness = "exact".into();
    spec.cost_tier = "T1".into();
    spec.wire_roles = wire_roles;
    spec
}

fn validate_unique_context<T: Eq + std::hash::Hash>(
    values: &[T],
    name: &str,
) -> Result<(), FeatureBuildError> {
    if values.is_empty() {
        return Err(FeatureBuildError(format!(
            "at least one {name} entry is required"
        )));
    }
    let mut seen = HashSet::new();
    seen.try_reserve(values.len())
        .map_err(|_| FeatureBuildError(format!("{name} validation allocation failed")))?;
    if values.iter().any(|value| !seen.insert(value)) {
        return Err(FeatureBuildError(format!("{name} must be unique")));
    }
    Ok(())
}

fn ensure_unconfigured(
    registry: &FeatureRegistry,
    ids: &[FeatureId],
) -> Result<(), FeatureBuildError> {
    let specs = registry.specs();
    for id in ids {
        if specs[id.index()].producer_version != "unconfigured" {
            return Err(FeatureBuildError(format!(
                "{} already has a configured provider",
                id
            )));
        }
    }
    Ok(())
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    digest.update((indices.len() as u64).to_le_bytes());
    for index in indices {
        digest.update(index.to_le_bytes());
    }
}

fn finish_digest(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
