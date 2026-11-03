//! Checked sizes, counts, context identity, and error-status helpers.
use super::{PaidBasisBlockContext, WIRE_ROLES};
use crate::features::FeatureComputation;
use crate::packet::FeatureStatus;
use crate::paid_basis::{PaidBasisError, PaidShearConfig};
use sha2::{Digest, Sha256};

pub(super) fn workspace_estimate(dimension: usize, groups: usize) -> Option<usize> {
    let pair_values = dimension.checked_mul(dimension)?.checked_mul(4)?;
    let pair_bytes = pair_values.checked_mul(std::mem::size_of::<f32>())?;
    let incidence = groups
        .checked_mul(dimension)?
        .checked_mul(WIRE_ROLES.len())?
        .checked_mul(std::mem::size_of::<f32>())?;
    let group_values = groups
        .checked_mul(19)?
        .checked_mul(std::mem::size_of::<f32>())?;
    pair_bytes.checked_add(incidence)?.checked_add(group_values)
}

pub(super) fn d_channels(ordinal: u8) -> usize {
    match ordinal {
        9 | 10 => 1,
        11 | 12 => 2,
        13 => 12,
        14 => 4,
        _ => unreachable!("D09-D14 channel requested"),
    }
}

pub(super) fn status_for_compile_error(error: &PaidBasisError) -> FeatureStatus {
    let message = error.to_string();
    if ["overflow", "allocation", "exceeds", "exhausted"]
        .iter()
        .any(|needle| message.contains(needle))
    {
        FeatureStatus::ResourceLimit
    } else {
        FeatureStatus::Unsupported
    }
}

pub(super) fn config_digest(contexts: &[PaidBasisBlockContext], config: PaidShearConfig) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"verified_paid_basis_d09_d14_v1");
    hash_field(&mut digest, &(config.local_width as u64).to_le_bytes());
    hash_field(&mut digest, &(config.sweeps as u64).to_le_bytes());
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.dimension.to_le_bytes());
        hash_indices(&mut digest, &context.source_wires);
        hash_indices(&mut digest, &context.target_wires);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

pub(super) fn hash_indices(digest: &mut Sha256, values: &[usize]) {
    hash_field(digest, &(values.len() as u64).to_le_bytes());
    for value in values {
        hash_field(digest, &(*value as u64).to_le_bytes());
    }
}

pub(super) fn sorted_in_range(values: &[usize], dimension: u32) -> bool {
    !values.is_empty()
        && values.iter().all(|value| *value < dimension as usize)
        && !values.windows(2).any(|pair| pair[0] >= pair[1])
}

pub(super) fn unavailable(status: FeatureStatus, reason: impl Into<String>) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}

pub(super) fn resource(reason: impl Into<String>) -> (FeatureStatus, String) {
    (FeatureStatus::ResourceLimit, reason.into())
}

pub(super) fn unsupported(reason: impl Into<String>) -> (FeatureStatus, String) {
    (FeatureStatus::Unsupported, reason.into())
}
