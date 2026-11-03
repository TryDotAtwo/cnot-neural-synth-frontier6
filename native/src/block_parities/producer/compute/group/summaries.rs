use super::super::super::gf2::rank_patterns;
use super::super::incidence::{append_incidence, IncidenceColumn};
use super::super::numeric::{exact_f32_integer, exact_f32_integer_u128};
use crate::block_parities::producer::patterns::PreparedBlock;

pub(super) fn append_rank_summary(
    block: &PreparedBlock<'_>,
    values: &mut Vec<f32>,
    incidence: &mut Vec<f32>,
    token_valid: &mut Vec<u8>,
) -> Result<(), &'static str> {
    let nonzero = block
        .patterns
        .iter()
        .filter(|pattern| !pattern.is_zero())
        .collect::<Vec<_>>();
    let rank = rank_patterns(&nonzero, block.context.sources.len())?;
    let total = nonzero
        .iter()
        .try_fold(0usize, |sum, pattern| {
            sum.checked_add(block.data.counts[*pattern])
        })
        .ok_or("nonzero pattern count overflow")?;
    let max_frequency = nonzero
        .iter()
        .map(|pattern| block.data.counts[*pattern])
        .max()
        .unwrap_or(0);
    let scalars = [
        rank,
        nonzero.len(),
        total,
        total - nonzero.len(),
        max_frequency,
    ];
    for scalar in scalars {
        let Some(value) = exact_f32_integer(scalar) else {
            return Err("block rank summary exceeds exact f32 range");
        };
        values.push(value);
    }
    append_incidence(
        block.context,
        &block.data.codes,
        None,
        IncidenceColumn::AllTargets,
        incidence,
    );
    token_valid.push(1);
    Ok(())
}

pub(super) fn append_parity_weight(
    block: &PreparedBlock<'_>,
    values: &mut Vec<f32>,
    incidence: &mut Vec<f32>,
    token_valid: &mut Vec<u8>,
) -> Result<(), &'static str> {
    let parity_weight = block
        .data
        .codes
        .iter()
        .flat_map(|pattern| &pattern.0)
        .map(|word| u128::from(word.count_ones()))
        .sum::<u128>();
    let Some(value) = exact_f32_integer_u128(parity_weight) else {
        return Err("block parity weight exceeds exact f32 range");
    };
    values.push(value);
    append_incidence(
        block.context,
        &block.data.codes,
        None,
        IncidenceColumn::AllTargets,
        incidence,
    );
    token_valid.push(1);
    Ok(())
}
