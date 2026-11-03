use super::super::incidence::{append_incidence, IncidenceColumn};
use super::super::numeric::exact_f32_integer;
use crate::block_parities::producer::patterns::PreparedBlock;

pub(super) fn append(
    ordinal: u8,
    block: &PreparedBlock<'_>,
    values: &mut Vec<f32>,
    incidence: &mut Vec<f32>,
    token_valid: &mut Vec<u8>,
) -> Result<(), &'static str> {
    for pattern in &block.patterns {
        let count = block.data.counts[pattern];
        let value = match ordinal {
            1 => count,
            2 => 1,
            4 if !pattern.is_zero() => count,
            4 => 0,
            _ => unreachable!(),
        };
        let Some(value) = exact_f32_integer(value) else {
            return Err("pattern frequency exceeds exact f32 range");
        };
        values.push(value);
        append_incidence(
            block.context,
            &block.data.codes,
            Some(pattern),
            IncidenceColumn::MatchingTargets(pattern),
            incidence,
        );
        token_valid.push(1);
    }
    Ok(())
}
