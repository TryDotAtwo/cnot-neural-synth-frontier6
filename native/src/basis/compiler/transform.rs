use super::super::{BasisChoiceKind, BasisSide, BasisTemplate};
use crate::model::ModelError;
use crate::state::CnotGate;

pub(super) struct BasisPreparation {
    pub(super) transformed: Vec<Vec<u8>>,
    pub(super) gates: Vec<CnotGate>,
}

pub(super) fn apply(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    bits: &[bool],
) -> Result<BasisPreparation, ModelError> {
    let mut transformed = block.to_vec();
    let mut preparation = Vec::new();

    for (choice, selected) in template.choices.iter().zip(bits) {
        if !selected {
            continue;
        }
        let physical_control = u32::try_from(template.group_wires()[choice.control])
            .map_err(|_| ModelError("basis control wire exceeds u32 range".into()))?;
        let physical_target = u32::try_from(template.group_wires()[choice.target])
            .map_err(|_| ModelError("basis target wire exceeds u32 range".into()))?;
        match choice.kind {
            BasisChoiceKind::Swap => {
                preparation.extend([
                    CnotGate {
                        control: physical_control,
                        target: physical_target,
                    },
                    CnotGate {
                        control: physical_target,
                        target: physical_control,
                    },
                    CnotGate {
                        control: physical_control,
                        target: physical_target,
                    },
                ]);
                match template.side {
                    BasisSide::Source => {
                        for row in &mut transformed {
                            row.swap(choice.control, choice.target);
                        }
                    }
                    BasisSide::Target => {
                        transformed.swap(choice.control, choice.target);
                    }
                }
            }
            BasisChoiceKind::Add => {
                preparation.push(CnotGate {
                    control: physical_control,
                    target: physical_target,
                });
                match template.side {
                    BasisSide::Source => {
                        for row in &mut transformed {
                            row[choice.control] ^= row[choice.target];
                        }
                    }
                    BasisSide::Target => {
                        let source_row = transformed[choice.control].clone();
                        for (value, source) in transformed[choice.target].iter_mut().zip(source_row)
                        {
                            *value ^= source;
                        }
                    }
                }
            }
        }
    }

    Ok(BasisPreparation {
        transformed,
        gates: preparation,
    })
}
