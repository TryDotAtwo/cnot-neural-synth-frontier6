//! Exact rank values for explicitly labeled wire cuts.
use super::helpers::unavailable;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::rank_features::contexts::LabeledCut;
use crate::rank_features::rank_math::projected_rank;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::state::MatrixState;
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
