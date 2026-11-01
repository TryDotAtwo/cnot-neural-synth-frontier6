//! Exact GF(2) rank features for explicit cuts, candidates, and physical actions.
//!
//! Contextual values are augmentation-only: none is invariant under arbitrary
//! wire relabeling. A cut, candidate, or physical action is never guessed.

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
use std::fmt;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LabeledCut {
    n: u32,
    a: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RestrictedIntersection {
    n: u32,
    left: Vec<u32>,
    right: Vec<u32>,
    coordinates: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RankIncrementProbe {
    n: u32,
    span_rows: Vec<u32>,
    coordinates: Vec<u32>,
    candidate_support: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LabeledCutError(pub String);

impl fmt::Display for LabeledCutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for LabeledCutError {}

impl LabeledCut {
    pub fn new(n: u32, a: Vec<u32>) -> Result<Self, LabeledCutError> {
        if n < 2 {
            return Err(LabeledCutError("cut width must be at least two".into()));
        }
        if a.is_empty() || a.len() >= n as usize {
            return Err(LabeledCutError(
                "cut A must be a nonempty proper subset of the wires".into(),
            ));
        }
        if a.iter().any(|wire| *wire >= n) {
            return Err(LabeledCutError("cut wire is outside the matrix".into()));
        }
        if a.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(LabeledCutError(
                "cut A must be sorted and contain no repeated wire".into(),
            ));
        }
        Ok(Self { n, a })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn side_a(&self) -> &[u32] {
        &self.a
    }

    pub fn side_b(&self) -> Vec<u32> {
        let mut b = Vec::with_capacity(self.n as usize - self.a.len());
        let mut next_a = 0usize;
        for wire in 0..self.n {
            if self.a.get(next_a).copied() == Some(wire) {
                next_a += 1;
            } else {
                b.push(wire);
            }
        }
        b
    }

    /// Relabels each selected wire by its source-to-destination image.
    pub fn relabel(&self, permutation: &[u32]) -> Result<Self, LabeledCutError> {
        let width = self.n as usize;
        if permutation.len() != width {
            return Err(LabeledCutError(
                "a full wire permutation is required".into(),
            ));
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(width)
            .map_err(|_| LabeledCutError("wire permutation validation allocation failed".into()))?;
        seen.resize(width, false);
        for &destination in permutation {
            let index = destination as usize;
            if index >= width || std::mem::replace(&mut seen[index], true) {
                return Err(LabeledCutError(
                    "wire relabeling must be a permutation of all wires".into(),
                ));
            }
        }
        let mut a = self
            .a
            .iter()
            .map(|source| permutation[*source as usize])
            .collect::<Vec<_>>();
        a.sort_unstable();
        Self::new(self.n, a)
    }
}

impl RestrictedIntersection {
    pub fn new(
        n: u32,
        left: Vec<u32>,
        right: Vec<u32>,
        coordinates: Vec<u32>,
    ) -> Result<Self, LabeledCutError> {
        if n == 0
            || !valid_sorted_indices(&left, n, false)
            || !valid_sorted_indices(&right, n, false)
            || !valid_sorted_indices(&coordinates, n, false)
        {
            return Err(LabeledCutError(
                "intersection probes need nonempty sorted unique in-range indices".into(),
            ));
        }
        Ok(Self {
            n,
            left,
            right,
            coordinates,
        })
    }
}

impl RankIncrementProbe {
    pub fn new(
        n: u32,
        span_rows: Vec<u32>,
        coordinates: Vec<u32>,
        candidate_support: Vec<u32>,
    ) -> Result<Self, LabeledCutError> {
        if n == 0
            || !valid_sorted_indices(&span_rows, n, false)
            || !valid_sorted_indices(&coordinates, n, false)
            || !valid_sorted_indices(&candidate_support, n, true)
        {
            return Err(LabeledCutError(
                "rank-increment probes need sorted unique in-range indices".into(),
            ));
        }
        Ok(Self {
            n,
            span_rows,
            coordinates,
            candidate_support,
        })
    }
}

fn valid_sorted_indices(values: &[u32], n: u32, allow_empty: bool) -> bool {
    (allow_empty || !values.is_empty())
        && values.iter().all(|value| *value < n)
        && !values.windows(2).any(|pair| pair[0] >= pair[1])
}

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
                return unavailable(FeatureStatus::Unsupported, "no_nonzero_partial_shear_block")
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

fn projected_rank(
    state: &MatrixState,
    selected_rows: &[u32],
    selected_columns: &[u32],
    subtract_identity: bool,
    transpose: bool,
    appended_candidate: Option<&[u32]>,
) -> Result<usize, &'static str> {
    let n = state.dimension() as usize;
    if selected_rows.iter().any(|row| *row as usize >= n)
        || selected_columns.iter().any(|column| *column as usize >= n)
        || selected_rows.is_empty()
        || selected_columns.is_empty()
    {
        return Err("labeled cut index is outside the state");
    }
    if appended_candidate.is_some_and(|support| support.iter().any(|wire| *wire as usize >= n)) {
        return Err("candidate support is outside the state");
    }
    let columns = selected_columns.len();
    let words = columns.checked_add(63).ok_or("cut matrix width overflow")? / 64;
    let row_count = selected_rows
        .len()
        .checked_add(usize::from(appended_candidate.is_some()))
        .ok_or("cut matrix row count overflow")?;
    let word_count = row_count
        .checked_mul(words)
        .ok_or("cut matrix storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "cut matrix workspace allocation failed")?;
    matrix.resize(word_count, 0u64);

    let packed = state.packed_rows();
    for (output_row, selected_row) in selected_rows.iter().copied().enumerate() {
        for (output_column, selected_column) in selected_columns.iter().copied().enumerate() {
            let (source_row, source_column) = if transpose {
                (selected_column as usize, selected_row as usize)
            } else {
                (selected_row as usize, selected_column as usize)
            };
            let source_word = source_row
                .checked_mul(state.words_per_row())
                .and_then(|base| base.checked_add(source_column / 64))
                .ok_or("source matrix index overflow")?;
            if source_word >= packed.len() {
                return Err("labeled cut index is outside the state");
            }
            let mut bit = (packed[source_word] >> (source_column % 64)) & 1;
            if subtract_identity && source_row == source_column {
                bit ^= 1;
            }
            if bit != 0 {
                matrix[output_row * words + output_column / 64] |= 1u64 << (output_column % 64);
            }
        }
    }

    if let Some(candidate_support) = appended_candidate {
        let output_row = selected_rows.len();
        let row_offset = output_row
            .checked_mul(words)
            .ok_or("candidate row index overflow")?;
        for (output_column, source_column) in selected_columns.iter().copied().enumerate() {
            if candidate_support.binary_search(&source_column).is_ok() {
                matrix[row_offset + output_column / 64] |= 1u64 << (output_column % 64);
            }
        }
    }

    Ok(eliminate_packed_rows(
        &mut matrix,
        row_count,
        columns,
        words,
    ))
}

fn rank_binary_block(block: &[Vec<u8>]) -> Result<usize, &'static str> {
    let Some(first_row) = block.first() else {
        return Err("partial-shear block has no target rows");
    };
    let columns = first_row.len();
    if columns == 0
        || block
            .iter()
            .any(|row| row.len() != columns || row.iter().any(|bit| *bit > 1))
    {
        return Err("partial-shear block has invalid binary shape");
    }
    let words = columns
        .checked_add(63)
        .ok_or("partial-shear width overflow")?
        / 64;
    let word_count = block
        .len()
        .checked_mul(words)
        .ok_or("partial-shear rank storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "partial-shear rank workspace allocation failed")?;
    matrix.resize(word_count, 0u64);
    for (row_index, row) in block.iter().enumerate() {
        for (column, bit) in row.iter().copied().enumerate() {
            if bit != 0 {
                matrix[row_index * words + column / 64] |= 1u64 << (column % 64);
            }
        }
    }
    Ok(eliminate_packed_rows(
        &mut matrix,
        block.len(),
        columns,
        words,
    ))
}

fn eliminate_packed_rows(
    matrix: &mut [u64],
    row_count: usize,
    columns: usize,
    words: usize,
) -> usize {
    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (rank..row_count).find(|row| matrix[*row * words + word] & bit != 0)
        else {
            continue;
        };
        if pivot != rank {
            for index in 0..words {
                matrix.swap(pivot * words + index, rank * words + index);
            }
        }
        for row in (rank + 1)..row_count {
            if matrix[row * words + word] & bit != 0 {
                for index in 0..words {
                    let pivot_value = matrix[rank * words + index];
                    matrix[row * words + index] ^= pivot_value;
                }
            }
        }
        rank += 1;
        if rank == row_count {
            break;
        }
    }
    rank
}
