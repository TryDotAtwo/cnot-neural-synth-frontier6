use super::GeneralCompilerError;
use crate::state::{CnotGate, MatrixState};

pub(super) fn to_identity(
    state: &MatrixState,
    n_usize: usize,
    max_gates: u64,
) -> Result<Vec<CnotGate>, GeneralCompilerError> {
    let mut reduced = state.clone();
    let mut reductions = Vec::new();

    for column in 0..n_usize {
        let pivot = (column..n_usize)
            .find(|row| bit_is_set(&reduced, *row, column))
            .ok_or_else(|| {
                GeneralCompilerError::Unsupported(
                    "matrix is singular under Gauss-Jordan elimination".into(),
                )
            })?;

        if pivot != column {
            let pivot = u32::try_from(pivot)
                .map_err(|_| GeneralCompilerError::ResourceLimit("pivot exceeds u32".into()))?;
            let column = u32::try_from(column)
                .map_err(|_| GeneralCompilerError::ResourceLimit("column exceeds u32".into()))?;
            for gate in [
                CnotGate {
                    control: pivot,
                    target: column,
                },
                CnotGate {
                    control: column,
                    target: pivot,
                },
                CnotGate {
                    control: pivot,
                    target: column,
                },
            ] {
                apply_reduction(&mut reduced, &mut reductions, gate, max_gates)?;
            }
        }

        let control = u32::try_from(column)
            .map_err(|_| GeneralCompilerError::ResourceLimit("column exceeds u32".into()))?;
        for row in 0..n_usize {
            if row == column || !bit_is_set(&reduced, row, column) {
                continue;
            }
            let target = u32::try_from(row)
                .map_err(|_| GeneralCompilerError::ResourceLimit("row exceeds u32".into()))?;
            apply_reduction(
                &mut reduced,
                &mut reductions,
                CnotGate { control, target },
                max_gates,
            )?;
        }
    }

    if !reduced.is_identity() {
        return Err(GeneralCompilerError::Invariant(
            "Gauss-Jordan reduction did not reach identity".into(),
        ));
    }

    Ok(reductions)
}

fn bit_is_set(state: &MatrixState, row: usize, column: usize) -> bool {
    let offset = row * state.words_per_row() + column / 64;
    (state.packed_rows()[offset] & (1u64 << (column % 64))) != 0
}

fn apply_reduction(
    state: &mut MatrixState,
    reductions: &mut Vec<CnotGate>,
    gate: CnotGate,
    max_gates: u64,
) -> Result<(), GeneralCompilerError> {
    if u64::try_from(reductions.len()).unwrap_or(u64::MAX) >= max_gates {
        return Err(GeneralCompilerError::ResourceLimit(format!(
            "compiler exceeded {max_gates} CNOT gates"
        )));
    }
    reductions
        .try_reserve(1)
        .map_err(|_| GeneralCompilerError::ResourceLimit("could not grow gate word".into()))?;
    state
        .apply_cnot(gate)
        .map_err(|error| GeneralCompilerError::Invariant(error.to_string()))?;
    reductions.push(gate);
    Ok(())
}
