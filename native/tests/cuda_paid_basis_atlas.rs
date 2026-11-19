use cnot103_native::device_paid_basis_atlas::DevicePaidBasisAtlas;
use cnot103_native::{optimize_exact_basis, BasisSide};
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    width: usize,
    side: String,
    vectors: Vec<Vec<u8>>,
    operations: Vec<(usize, usize)>,
    direct_cost: usize,
    paid_cost: usize,
    transformed_vectors: Vec<Vec<u8>>,
}
fn integers(t: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).flatten(0, -1)).unwrap()
}
fn check(width: usize, side: BasisSide, batches: &[Vec<Vec<u8>>]) {
    let atlas = DevicePaidBasisAtlas::new(width, side, 0).unwrap();
    let rows = batches[0].len();
    let values = batches
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let vectors = Tensor::from_slice(&values)
        .view([batches.len() as i64, rows as i64, width as i64])
        .to_device(Device::Cuda(0));
    let result = atlas.optimize(&vectors).unwrap();
    assert_eq!(
        integers(&atlas.state_count()),
        vec![[0, 1, 6, 168, 20160][width]]
    );
    assert_eq!(integers(result.status()), vec![1; batches.len()]);
    let lengths = integers(result.lengths());
    let operations = integers(result.operations());
    let transformed = integers(result.transformed());
    let direct = integers(result.direct_cost());
    let paid = integers(result.paid_cost());
    let capacity = result.operations().size()[1] as usize;
    for (batch, input) in batches.iter().enumerate() {
        let expected = optimize_exact_basis(input, side).unwrap();
        assert_eq!(lengths[batch] as usize, expected.operations.len());
        assert_eq!(direct[batch] as usize, expected.direct_cost);
        assert_eq!(paid[batch] as usize, expected.paid_cost);
        let output_ops = &operations[batch * capacity * 2..(batch + 1) * capacity * 2];
        let expected_ops = expected
            .operations
            .iter()
            .flat_map(|&(a, b)| [a as i64, b as i64])
            .collect::<Vec<_>>();
        assert_eq!(&output_ops[..expected_ops.len()], expected_ops);
        assert!(output_ops[expected_ops.len()..].iter().all(|&v| v == 0));
        assert_eq!(
            &transformed[batch * rows * width..(batch + 1) * rows * width],
            expected
                .transformed_vectors
                .iter()
                .flatten()
                .map(|&v| i64::from(v))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn gpu_paid_basis_atlas_matches_pinned_source_words_and_full_transformed_vectors() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/exact_paid_basis_reference_v9.json")).unwrap();
    assert_eq!(fixture.cases.len(), 6);
    for case in fixture.cases {
        let side = if case.side == "source" {
            BasisSide::Source
        } else {
            BasisSide::Target
        };
        let reference = optimize_exact_basis(&case.vectors, side).unwrap();
        assert_eq!(reference.operations, case.operations);
        assert_eq!(reference.direct_cost, case.direct_cost);
        assert_eq!(reference.paid_cost, case.paid_cost);
        assert_eq!(reference.transformed_vectors, case.transformed_vectors);
        check(case.width, side, &[case.vectors]);
    }
}

#[test]
fn gpu_paid_basis_atlas_preserves_first_minimum_across_histograms_and_both_sides() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    for side in [BasisSide::Source, BasisSide::Target] {
        check(1, side, &[vec![vec![0]; 17], vec![vec![1]; 17]]);
        let batches = (0..256)
            .map(|pattern| {
                let mut rows = Vec::new();
                for code in 0..4 {
                    for _ in 0..((pattern >> (2 * code)) & 3) {
                        rows.push(vec![(code & 1) as u8, ((code >> 1) & 1) as u8]);
                    }
                }
                rows.resize(12, vec![0, 0]);
                rows
            })
            .collect::<Vec<_>>();
        check(2, side, &batches);
        for width in [3, 4] {
            let batches = (0..32)
                .map(|sample| {
                    (0..73)
                        .map(|row| {
                            // Deterministic, repeated patterns and zero rows; no RNG state.
                            let code = (sample * 97 + row * 31 + (row / 7) * sample) % (1 << width);
                            (0..width)
                                .map(|bit| ((code >> bit) & 1) as u8)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            check(width, side, &batches);
            check(width, side, &[vec![vec![0; width]; 9]]);
        }
    }
}

#[test]
fn gpu_paid_basis_atlas_checks_input_metadata_and_invalid_bits_without_partial_outputs() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    assert!(DevicePaidBasisAtlas::new(0, BasisSide::Source, 0).is_err());
    assert!(DevicePaidBasisAtlas::new(5, BasisSide::Source, 0).is_err());
    let atlas = DevicePaidBasisAtlas::new(3, BasisSide::Target, 0).unwrap();
    // An inspection tensor must not expose mutable aliasing of internal
    // traversal metadata used by subsequent unsafe CUDA launches.
    let mut count = atlas.state_count();
    let _ = count.fill_(0);
    assert_eq!(integers(&atlas.state_count()), [168]);
    assert!(atlas
        .optimize(&Tensor::zeros([1, 2, 3], (Kind::Uint8, Device::Cpu)))
        .is_err());
    assert!(atlas
        .optimize(&Tensor::zeros([1, 0, 3], (Kind::Uint8, Device::Cuda(0))))
        .is_err());
    assert!(atlas
        .optimize(&Tensor::zeros([0, 2, 3], (Kind::Uint8, Device::Cuda(0))))
        .is_err());
    assert!(atlas
        .optimize(&Tensor::zeros([1, 2, 2], (Kind::Uint8, Device::Cuda(0))))
        .is_err());
    assert!(atlas
        .optimize(&Tensor::zeros([1, 2, 3], (Kind::Int, Device::Cuda(0))))
        .is_err());
    let vectors = Tensor::from_slice(&[1u8, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1])
        .view([2, 2, 3])
        .to_device(Device::Cuda(0));
    let result = atlas.optimize(&vectors).unwrap();
    assert_eq!(integers(result.status()), [3, 1]);
    assert_eq!(integers(result.lengths())[0], 0);
    assert_eq!(integers(result.direct_cost())[0], 0);
    assert_eq!(integers(result.paid_cost())[0], 0);
    assert!(integers(&result.operations().get(0))
        .iter()
        .all(|&x| x == 0));
    assert!(integers(&result.transformed().get(0))
        .iter()
        .all(|&x| x == 0));
}
