//! Construct the theorem-selected encoder word from its classified frame.

use crate::state::CnotGate;

use super::super::math::{gate, inverse_2x2, multiply_2x2, IDENTITY_FRAME};
use super::super::types::{Stage35Branch, Stage35Error};

pub(super) fn encode_word(
    branch: Stage35Branch,
    frame: &[(usize, usize)],
    formula_cost: usize,
    active: [usize; 2],
    columns: &[(usize, u8)],
    transformed: &[(usize, u8)],
    shared: &[usize],
) -> Result<Vec<CnotGate>, Stage35Error> {
    let mut encoder_gates = Vec::with_capacity(formula_cost);
    match branch {
        Stage35Branch::NoFrameMotion => {
            for (column, value) in columns.iter().copied() {
                if value & 1 != 0 {
                    encoder_gates.push(gate(column, active[0])?);
                }
                if value & 2 != 0 {
                    encoder_gates.push(gate(column, active[1])?);
                }
            }
        }
        Stage35Branch::NonemptyFrameMotion => {
            for (column, value) in transformed {
                if *value == 1 || *value == 2 {
                    let target = if *value == 1 { active[0] } else { active[1] };
                    encoder_gates.push(gate(*column, target)?);
                }
            }
            let (control, target) = frame[0];
            encoder_gates.push(gate(active[control], active[target])?);

            let mut first_frame = IDENTITY_FRAME;
            first_frame[target] ^= first_frame[control];
            let first_inverse = inverse_2x2(first_frame)?;
            let receiver = (0..2)
                .find(|row| multiply_2x2(first_inverse, 1u8 << row) == 3)
                .ok_or(Stage35Error::MissingFrame(first_frame))?;
            for column in shared {
                encoder_gates.push(gate(*column, active[receiver])?);
            }
            for (control, target) in frame.iter().copied().skip(1) {
                encoder_gates.push(gate(active[control], active[target])?);
            }
        }
    }
    Ok(encoder_gates)
}
