//! Exact paid-basis compilation into a replayable CNOT word certificate.

use super::{BasisChoiceKind, BasisTemplate};
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::model::ModelError;
use crate::state::{CnotGate, MatrixState};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasisCandidate {
    pub certificate: WordCertificate,
    pub preparation: Vec<CnotGate>,
    pub core: Vec<CnotGate>,
    pub bits: Vec<bool>,
    pub transformed_block: Vec<Vec<u8>>,
}

pub fn compile_basis(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    bits: &[bool],
) -> Result<BasisCandidate, ModelError> {
    template.validate()?;
    validate_block(template, block)?;
    if bits.len() != template.choices.len() {
        return Err(ModelError(
            "basis assignment width differs from template".into(),
        ));
    }
    let n = u32::try_from(template.n)
        .map_err(|_| ModelError("basis width exceeds CNOT state dimension range".into()))?;
    let alphabet =
        CnotAlphabet::all_to_all_unit(n).map_err(|error| ModelError(error.to_string()))?;
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
                    super::BasisSide::Source => {
                        for row in &mut transformed {
                            row.swap(choice.control, choice.target);
                        }
                    }
                    super::BasisSide::Target => {
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
                    super::BasisSide::Source => {
                        for row in &mut transformed {
                            row[choice.control] ^= row[choice.target];
                        }
                    }
                    super::BasisSide::Target => {
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

    let mut core = Vec::new();
    for (target_index, row) in transformed.iter().enumerate() {
        let target = u32::try_from(template.target_wires[target_index])
            .map_err(|_| ModelError("basis target wire exceeds u32 range".into()))?;
        for (source_index, bit) in row.iter().copied().enumerate() {
            if bit != 0 {
                let control = u32::try_from(template.source_wires[source_index])
                    .map_err(|_| ModelError("basis source wire exceeds u32 range".into()))?;
                core.push(CnotGate { control, target });
            }
        }
    }

    let mut gates = Vec::with_capacity(2 * preparation.len() + core.len());
    gates.extend(preparation.iter().copied());
    gates.extend(core.iter().copied());
    gates.extend(preparation.iter().rev().copied());
    let mut endpoint =
        MatrixState::identity(n, alphabet.id()).map_err(|error| ModelError(error.to_string()))?;
    for gate in &gates {
        endpoint
            .apply_cnot(*gate)
            .map_err(|error| ModelError(error.to_string()))?;
    }

    let start =
        MatrixState::identity(n, alphabet.id()).map_err(|error| ModelError(error.to_string()))?;
    let mut expected_rows = start.packed_rows().to_vec();
    let words_per_row = start.words_per_row();
    for (target_index, &target_wire) in template.target_wires.iter().enumerate() {
        for (source_index, &source_wire) in template.source_wires.iter().enumerate() {
            if block[target_index][source_index] != 0 {
                let offset = target_wire * words_per_row + source_wire / 64;
                expected_rows[offset] ^= 1u64 << (source_wire % 64);
            }
        }
    }
    let expected = MatrixState::from_packed_rows(n, alphabet.id(), expected_rows)
        .map_err(|error| ModelError(error.to_string()))?;
    if endpoint != expected {
        return Err(ModelError(
            "compiled basis word does not realize the declared full shear".into(),
        ));
    }

    let physical_cost = u64::try_from(gates.len())
        .map_err(|_| ModelError("compiled basis word length exceeds u64".into()))?;
    let certificate = WordCertificate {
        start,
        end: endpoint,
        gates,
        physical_cost,
        producer: "own_J17_template_v1".into(),
        lineage_id: "basis_template".into(),
    };
    certificate
        .verify(&alphabet)
        .map_err(|error| ModelError(error.to_string()))?;

    Ok(BasisCandidate {
        certificate,
        preparation,
        core,
        bits: bits.to_vec(),
        transformed_block: transformed,
    })
}

fn validate_block(template: &BasisTemplate, block: &[Vec<u8>]) -> Result<(), ModelError> {
    if block.len() != template.target_wires.len()
        || block
            .iter()
            .any(|row| row.len() != template.source_wires.len() || row.iter().any(|bit| *bit > 1))
    {
        return Err(ModelError(
            "basis block must be a binary target-by-source matrix".into(),
        ));
    }
    Ok(())
}
