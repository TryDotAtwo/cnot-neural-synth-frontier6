use cnot103_native::{optimize_exact_basis, BasisSide};
use serde::Deserialize;

const SOURCE_FIXTURE: &str = include_str!("fixtures/exact_paid_basis_reference_v9.json");

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    reference_file: String,
    reference_source_sha256: String,
    algorithm: String,
    cases: Vec<GoldenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    width: usize,
    side: String,
    vectors: Vec<Vec<u8>>,
    operations: Vec<(usize, usize)>,
    direct_cost: usize,
    paid_cost: usize,
    transformed_vectors: Vec<Vec<u8>>,
}

#[test]
fn exact_paid_basis_matches_source_python_golden_cases() {
    let fixture: GoldenFixture = serde_json::from_str(SOURCE_FIXTURE).unwrap();
    assert_eq!(fixture.reference_file, "code/exact_basis.py");
    assert_eq!(
        fixture.reference_source_sha256,
        "96b826d1d1d0e91f14ffd5349069254335521b505596059026eeb1da77db424d"
    );
    assert_eq!(
        fixture.algorithm,
        "atlas(width, side) followed by optimize(vectors, side)"
    );
    assert_eq!(fixture.cases.len(), 6);

    for case in fixture.cases {
        assert!(case.vectors.iter().all(|row| row.len() == case.width));
        let side = match case.side.as_str() {
            "source" => BasisSide::Source,
            "target" => BasisSide::Target,
            other => panic!("unknown golden side: {other}"),
        };
        let result = optimize_exact_basis(&case.vectors, side).unwrap();

        assert_eq!(
            result.operations, case.operations,
            "{} basis tie-break",
            case.side
        );
        assert_eq!(
            result.direct_cost, case.direct_cost,
            "{} direct cost",
            case.side
        );
        assert_eq!(result.paid_cost, case.paid_cost, "{} paid cost", case.side);
        assert_eq!(
            result.transformed_vectors, case.transformed_vectors,
            "{} transformed block",
            case.side
        );
        assert_eq!(
            2 * result.operations.len()
                + result
                    .transformed_vectors
                    .iter()
                    .flatten()
                    .map(|bit| usize::from(*bit))
                    .sum::<usize>(),
            result.paid_cost,
            "preparation and restoration must both be paid"
        );
        assert!(result.paid_cost <= result.direct_cost);
    }
}

#[test]
fn exact_paid_basis_rejects_malformed_or_unsupported_blocks() {
    assert!(optimize_exact_basis(&[], BasisSide::Source).is_err());
    assert!(optimize_exact_basis(&[vec![0, 2]], BasisSide::Source).is_err());
    assert!(optimize_exact_basis(&[vec![0, 1], vec![1]], BasisSide::Target).is_err());
    assert!(optimize_exact_basis(&[vec![0, 0, 0, 0, 0]], BasisSide::Source).is_err());
}
