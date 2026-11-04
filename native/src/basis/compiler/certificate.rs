use super::super::BasisTemplate;
use super::transform::BasisPreparation;
use super::BasisCandidate;
use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::model::ModelError;
use crate::state::{CnotGate, MatrixState};

pub(super) fn build(
    template: &BasisTemplate,
    block: &[Vec<u8>],
    bits: &[bool],
    n: u32,
    alphabet: &CnotAlphabet,
    prepared: BasisPreparation,
) -> Result<BasisCandidate, ModelError> {
    let BasisPreparation {
        transformed,
        gates: preparation,
    } = prepared;
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
        .verify(alphabet)
        .map_err(|error| ModelError(error.to_string()))?;

    Ok(BasisCandidate {
        certificate,
        preparation,
        core,
        bits: bits.to_vec(),
        transformed_block: transformed,
    })
}
