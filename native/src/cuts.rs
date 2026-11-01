//! Exact GF(2) rank features for caller-specified labeled wire cuts.
//!
//! C02--C04 are augmentation-only: their values depend on a physical wire
//! partition and therefore are not invariant under arbitrary wire relabeling.
//! No cut is inferred from wire indices or from the matrix itself.

use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LabeledCut {
    n: u32,
    a: Vec<u32>,
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
            let rank = match projected_rank(state, rows, columns, subtract_identity) {
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
) -> Result<usize, &'static str> {
    let n = state.dimension() as usize;
    if selected_rows.iter().any(|row| *row as usize >= n)
        || selected_columns.iter().any(|column| *column as usize >= n)
        || selected_rows.is_empty()
        || selected_columns.is_empty()
    {
        return Err("labeled cut index is outside the state");
    }
    let columns = selected_columns.len();
    let words = columns.checked_add(63).ok_or("cut matrix width overflow")? / 64;
    let word_count = selected_rows
        .len()
        .checked_mul(words)
        .ok_or("cut matrix storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "cut matrix workspace allocation failed")?;
    matrix.resize(word_count, 0u64);

    let packed = state.packed_rows();
    for (output_row, source_row) in selected_rows.iter().copied().enumerate() {
        for (output_column, source_column) in selected_columns.iter().copied().enumerate() {
            let source_row = source_row as usize;
            let source_column = source_column as usize;
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

    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) =
            (rank..selected_rows.len()).find(|row| matrix[*row * words + word] & bit != 0)
        else {
            continue;
        };
        if pivot != rank {
            for index in 0..words {
                matrix.swap(pivot * words + index, rank * words + index);
            }
        }
        for row in (rank + 1)..selected_rows.len() {
            if matrix[row * words + word] & bit != 0 {
                for index in 0..words {
                    let pivot_value = matrix[rank * words + index];
                    matrix[row * words + index] ^= pivot_value;
                }
            }
        }
        rank += 1;
        if rank == selected_rows.len() {
            break;
        }
    }
    Ok(rank)
}
