use super::super::incidence::{append_incidence, IncidenceColumn};
use super::super::numeric::exact_f32_integer;
use crate::block_parities::producer::patterns::PreparedBlock;

pub(super) fn append(
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
    for left_index in 0..nonzero.len() {
        for right_index in (left_index + 1)..nonzero.len() {
            let left = nonzero[left_index];
            let right = nonzero[right_index];
            let Some(value) = exact_f32_integer(left.intersection_count(right)) else {
                return Err("pattern intersection exceeds exact f32 range");
            };
            values.push(value);
            append_incidence(
                block.context,
                &block.data.codes,
                Some(left),
                IncidenceColumn::SourcePattern(right),
                incidence,
            );
            token_valid.push(1);
        }
    }
    Ok(())
}
