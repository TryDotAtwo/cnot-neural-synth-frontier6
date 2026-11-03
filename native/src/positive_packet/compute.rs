use self::storage::zeroed_f32;
use super::{OptionalGatePacket, MAX_EXACT_F32_INTEGER};
use crate::features::FeatureValue;
use crate::state::MatrixState;

mod budget;
mod polynomial;
mod rank;
mod storage;

pub(super) use budget::{factor_entry_count, working_bytes_upper_bound};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ComputeFailure {
    ResourceLimit,
    NonAffine,
    Internal,
}

pub(super) fn compute_values(
    state: &MatrixState,
    context: &OptionalGatePacket,
    max_monomials: usize,
) -> Result<FeatureValue, ComputeFailure> {
    let n = usize::try_from(context.n).map_err(|_| ComputeFailure::ResourceLimit)?;
    let slots = context.gates.len();
    let columns = polynomial::affine_columns(context, max_monomials)?;
    let prefix = rank::incremental_ranks(&columns, false)?;
    let suffix = rank::incremental_ranks(&columns, true)?;
    let total_rank = *prefix.last().ok_or(ComputeFailure::Internal)?;
    let groups = slots.checked_add(1).ok_or(ComputeFailure::ResourceLimit)?;
    let value_count = groups.checked_mul(4).ok_or(ComputeFailure::ResourceLimit)?;
    let incidence_count = groups
        .checked_mul(n)
        .and_then(|count| count.checked_mul(2))
        .ok_or(ComputeFailure::ResourceLimit)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    let mut incidence = zeroed_f32(incidence_count)?;
    let mut token_valid = Vec::new();
    token_valid
        .try_reserve_exact(groups)
        .map_err(|_| ComputeFailure::ResourceLimit)?;
    token_valid.resize(groups, 1);

    for cut in 0..groups {
        let width = prefix[cut]
            .checked_add(suffix[cut])
            .and_then(|sum| sum.checked_sub(total_rank))
            .ok_or(ComputeFailure::Internal)?;
        let mut live_slots = Vec::new();
        live_slots
            .try_reserve_exact(cut)
            .map_err(|_| ComputeFailure::ResourceLimit)?;
        for slot in 0..cut {
            let live = context.factors.iter().any(|factor| {
                factor.scope.iter().copied().max().is_some_and(|last| {
                    usize::try_from(last).is_ok_and(|last| last >= cut)
                        && factor.scope.contains(&(slot as u32))
                })
            });
            if live {
                live_slots.push(slot);
                let (control, target) = context.gates[slot];
                let control = usize::try_from(control).map_err(|_| ComputeFailure::Internal)?;
                let target = usize::try_from(target).map_err(|_| ComputeFailure::Internal)?;
                incidence[(cut * n + control) * 2] = 1.0;
                incidence[(cut * n + target) * 2 + 1] = 1.0;
            }
        }
        let live = live_slots.len();
        let upper = width.checked_add(live).ok_or(ComputeFailure::Internal)?;
        if cut > MAX_EXACT_F32_INTEGER || upper > MAX_EXACT_F32_INTEGER {
            return Err(ComputeFailure::ResourceLimit);
        }
        values.extend([cut as f32, width as f32, live as f32, upper as f32]);
    }

    // Keep the state argument in the exact contract even though the displayed
    // widths are independent of its RHS.
    if state.dimension() != context.n {
        return Err(ComputeFailure::Internal);
    }
    Ok(FeatureValue {
        shape: vec![groups, 4],
        values,
        token_valid,
        incidence,
    })
}
