use cnot103_native::{
    compile_paid_shear, BasisSide, CnotAlphabet, CnotGate, MatrixState, PaidShearConfig,
};
use serde::Deserialize;

const SOURCE_FIXTURE: &str = include_str!("fixtures/paid_shear_reference_v9.json");

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    compiler_file: String,
    compiler_source_sha256: String,
    local_atlas_file: String,
    local_atlas_source_sha256: String,
    block: Vec<Vec<u8>>,
    source_wires: Vec<usize>,
    target_wires: Vec<usize>,
    dimension: usize,
    direct_cost: usize,
    paid_cost: usize,
    preparation: Vec<(u32, u32)>,
    restoration: Vec<(u32, u32)>,
    core: Vec<(u32, u32)>,
    word: Vec<(u32, u32)>,
    transformed_block: Vec<Vec<u8>>,
    source_basis: Vec<Vec<u8>>,
    target_basis: Vec<Vec<u8>>,
    tested_groups: usize,
    accepted_moves: Vec<GoldenMove>,
}

#[derive(Debug, Deserialize)]
struct GoldenMove {
    sweep: usize,
    side: String,
    group_indices: Vec<usize>,
    cost_before: usize,
    cost_after: usize,
}

#[test]
fn paid_shear_compiler_matches_source_and_verifies_full_endpoint() {
    let fixture: GoldenFixture = serde_json::from_str(SOURCE_FIXTURE).unwrap();
    assert_eq!(fixture.compiler_file, "code/shared_synth.py");
    assert_eq!(
        fixture.compiler_source_sha256,
        "c2e0d974d3ea44451ac2e79d0fd89f20429aa80e5eb88790e22999ea3a1fed3a"
    );
    assert_eq!(fixture.local_atlas_file, "code/exact_basis.py");
    assert_eq!(
        fixture.local_atlas_source_sha256,
        "96b826d1d1d0e91f14ffd5349069254335521b505596059026eeb1da77db424d"
    );

    let candidate = compile_paid_shear(
        &fixture.block,
        fixture.dimension,
        &fixture.source_wires,
        &fixture.target_wires,
        PaidShearConfig::source_exact(),
    )
    .unwrap();

    assert_eq!(candidate.direct_cost, fixture.direct_cost);
    assert_eq!(candidate.paid_cost, fixture.paid_cost);
    assert_eq!(gate_pairs(&candidate.preparation), fixture.preparation);
    assert_eq!(gate_pairs(&candidate.restoration), fixture.restoration);
    assert_eq!(gate_pairs(&candidate.core), fixture.core);
    assert_eq!(gate_pairs(&candidate.certificate.gates), fixture.word);
    assert_eq!(candidate.transformed_block, fixture.transformed_block);
    assert_eq!(candidate.source_basis, fixture.source_basis);
    assert_eq!(candidate.target_basis, fixture.target_basis);
    assert_eq!(candidate.tested_groups, fixture.tested_groups);

    assert_eq!(candidate.accepted_moves.len(), fixture.accepted_moves.len());
    for (actual, expected) in candidate.accepted_moves.iter().zip(&fixture.accepted_moves) {
        assert_eq!(actual.sweep, expected.sweep);
        assert_eq!(actual.side, BasisSide::Source);
        assert_eq!(actual.side.to_string(), expected.side);
        assert_eq!(actual.group_indices, expected.group_indices);
        assert_eq!(actual.cost_before, expected.cost_before);
        assert_eq!(actual.cost_after, expected.cost_after);
    }

    let alphabet = CnotAlphabet::all_to_all_unit(fixture.dimension as u32).unwrap();
    candidate.certificate.verify(&alphabet).unwrap();
    let mut expected = MatrixState::identity(fixture.dimension as u32, alphabet.id()).unwrap();
    for (target_index, target_wire) in fixture.target_wires.iter().copied().enumerate() {
        for (source_index, source_wire) in fixture.source_wires.iter().copied().enumerate() {
            if fixture.block[target_index][source_index] != 0 {
                expected
                    .apply_cnot(CnotGate {
                        control: source_wire as u32,
                        target: target_wire as u32,
                    })
                    .unwrap();
            }
        }
    }
    assert_eq!(candidate.certificate.end, expected);
}

#[test]
fn paid_shear_compiler_rejects_nonphysical_or_malformed_contexts() {
    let config = PaidShearConfig::source_exact();
    assert!(compile_paid_shear(&[], 4, &[0, 1], &[2, 3], config).is_err());
    assert!(compile_paid_shear(&[vec![1]], 3, &[0, 0], &[1], config).is_err());
    assert!(compile_paid_shear(&[vec![1]], 3, &[0], &[0], config).is_err());
    assert!(compile_paid_shear(&[vec![1]], 3, &[0], &[3], config).is_err());
    assert!(compile_paid_shear(&[vec![2]], 3, &[0], &[1], config).is_err());
}

fn gate_pairs(gates: &[CnotGate]) -> Vec<(u32, u32)> {
    gates
        .iter()
        .map(|gate| (gate.control, gate.target))
        .collect()
}
