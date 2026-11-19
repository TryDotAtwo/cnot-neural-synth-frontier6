use cnot103_native::device_paid_shear::DevicePaidShearPlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::{
    compile_paid_shear, BasisSide, CnotAlphabet, CnotGate, MatrixState, PaidBasisBlockContext,
    PaidShearConfig,
};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

fn ints(t: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).flatten(0, -1)).unwrap()
}
fn tensor_states(states: &[MatrixState], alphabet: &CnotAlphabet) -> DeviceStateBatch {
    let values = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect::<Vec<_>>();
    let n = alphabet.dimension();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&values)
            .view([states.len() as i64, n as i64, n.div_ceil(64) as i64])
            .to_device(Device::Cuda(0)),
        alphabet,
        0,
    )
    .unwrap()
}
fn state_for(block: &[Vec<u8>], n: u32, source: &[usize], target: &[usize]) -> MatrixState {
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let mut state = MatrixState::identity(n, alphabet.id()).unwrap();
    for (r, row) in block.iter().enumerate() {
        for (c, &value) in row.iter().enumerate() {
            if value != 0 {
                state
                    .apply_cnot(CnotGate {
                        control: source[c] as u32,
                        target: target[r] as u32,
                    })
                    .unwrap();
            }
        }
    }
    state
}
fn pairs(t: &Tensor, batch: usize, count: usize) -> Vec<CnotGate> {
    let values = ints(&t.get(batch as i64));
    assert!(
        values[count * 2..].iter().all(|&v| v == 0),
        "unused gates must be zero"
    );
    values[..count * 2]
        .chunks_exact(2)
        .map(|p| CnotGate {
            control: p[0] as u32,
            target: p[1] as u32,
        })
        .collect()
}
fn changed(matrix: &[Vec<u8>]) -> usize {
    matrix
        .iter()
        .enumerate()
        .map(|(r, row)| {
            row.iter()
                .enumerate()
                .filter(|(c, v)| **v != u8::from(r == *c))
                .count()
        })
        .sum()
}
fn check(
    states: &[MatrixState],
    source: &[usize],
    target: &[usize],
    config: PaidShearConfig,
) -> (usize, usize, usize) {
    let n = states[0].dimension();
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let plan = DevicePaidShearPlan::new(
        PaidBasisBlockContext::new(n, source.to_vec(), target.to_vec()).unwrap(),
        config,
        0,
    )
    .unwrap();
    let resident = tensor_states(states, &alphabet);
    let result = plan.compile(&resident).unwrap();
    assert_eq!(ints(result.status()), vec![1; states.len()]);
    let statistics = ints(result.statistics());
    let lengths = ints(result.lengths());
    let moves = ints(result.accepted_moves());
    let move_capacity = result.accepted_moves().size()[1] as usize;
    let mut totals = (0, 0, 0);
    for (batch, state) in states.iter().enumerate() {
        let block = target
            .iter()
            .map(|&r| {
                source
                    .iter()
                    .map(|&c| {
                        ((state.packed_rows()[r * n.div_ceil(64) as usize + c / 64] >> (c % 64))
                            & 1) as u8
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = compile_paid_shear(&block, n as usize, source, target, config).unwrap();
        let s = &statistics[batch * 16..(batch + 1) * 16];
        let source_moves = expected
            .accepted_moves
            .iter()
            .filter(|m| m.side == BasisSide::Source)
            .count();
        let target_moves = expected.accepted_moves.len() - source_moves;
        totals.0 += source_moves;
        totals.1 += target_moves;
        totals.2 += expected
            .accepted_moves
            .iter()
            .filter(|m| m.sweep > 0)
            .count();
        let cancels = expected
            .boundary_cancellations
            .iter()
            .map(|v| v.canceled_pairs)
            .sum::<usize>();
        assert_eq!(
            &s[..13],
            &[
                expected.direct_cost as i64,
                expected.paid_cost as i64,
                expected.preparation.len() as i64,
                expected.source_preparation.len() as i64,
                expected.target_preparation.len() as i64,
                expected.core.len() as i64,
                expected.restoration.len() as i64,
                expected.tested_groups as i64,
                expected.accepted_moves.len() as i64,
                source_moves as i64,
                target_moves as i64,
                changed(&expected.source_basis) as i64,
                changed(&expected.target_basis) as i64
            ]
        );
        assert_eq!(s[14], cancels as i64);
        assert_eq!(
            s[15],
            (2 * expected.preparation.len() + expected.core.len()) as i64
        );
        assert_eq!(lengths[batch], expected.paid_cost as i64);
        assert_eq!(
            pairs(result.preparation(), batch, expected.preparation.len()),
            expected.preparation
        );
        assert_eq!(
            pairs(result.restoration(), batch, expected.restoration.len()),
            expected.restoration
        );
        assert_eq!(
            pairs(result.core(), batch, expected.core.len()),
            expected.core
        );
        assert_eq!(
            pairs(result.word(), batch, expected.paid_cost),
            expected.certificate.gates
        );
        for (actual, expected_matrix) in [
            (result.transformed_block(), &expected.transformed_block),
            (result.source_basis(), &expected.source_basis),
            (result.target_basis(), &expected.target_basis),
        ] {
            assert_eq!(
                ints(&actual.get(batch as i64)),
                expected_matrix
                    .iter()
                    .flatten()
                    .map(|&v| i64::from(v))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            ints(&result.endpoint().get(batch as i64)),
            expected
                .certificate
                .end
                .packed_rows()
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<_>>()
        );
        let mut events = BTreeMap::new();
        for gate in pairs(result.cancellation_events(), batch, cancels) {
            *events.entry((gate.control, gate.target)).or_insert(0usize) += 1;
        }
        assert_eq!(
            events,
            expected
                .boundary_cancellations
                .iter()
                .map(|c| ((c.control, c.target), c.canceled_pairs))
                .collect()
        );
        for (index, movement) in expected.accepted_moves.iter().enumerate() {
            let mut expected_move = [0i64; 11];
            expected_move[0] = movement.sweep as i64;
            expected_move[1] = i64::from(movement.side == BasisSide::Target);
            expected_move[2] = movement.group_indices.len() as i64;
            for (i, &wire) in movement.group_indices.iter().enumerate() {
                expected_move[3 + i] = wire as i64;
            }
            expected_move[7] = movement.cost_before as i64;
            expected_move[8] = movement.cost_after as i64;
            expected_move[9] = movement.preparation_range.start as i64;
            expected_move[10] = movement.preparation_range.end as i64;
            let offset = (batch * move_capacity + index) * 11;
            assert_eq!(&moves[offset..offset + 11], expected_move);
        }
        let unused = (batch * move_capacity + expected.accepted_moves.len()) * 11;
        assert!(moves[unused..(batch + 1) * move_capacity * 11]
            .iter()
            .all(|&v| v == 0));
        // Independent full physical replay: compare against the shear target,
        // including untouched rows, not only the local transformed block.
        let mut replay = MatrixState::identity(n, alphabet.id()).unwrap();
        for gate in pairs(result.word(), batch, expected.paid_cost) {
            replay.apply_cnot(gate).unwrap();
        }
        assert_eq!(replay, expected.certificate.end);
    }
    let endpoints = DeviceStateBatch::from_packed_tensor_for_alphabet(
        result.endpoint().shallow_clone(),
        &alphabet,
        0,
    )
    .unwrap();
    assert_eq!(
        ints(
            &cnot103_native::device_compiler::verify_device_cnot_words(
                &endpoints,
                result.word(),
                result.lengths()
            )
            .unwrap()
        ),
        vec![1; states.len()]
    );
    totals
}

#[test]
fn gpu_paid_shear_matches_full_source_fixture_and_wide_physical_endpoints() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/paid_shear_reference_v9.json")).unwrap();
    let block: Vec<Vec<u8>> = serde_json::from_value(fixture["block"].clone()).unwrap();
    for (n, source, target) in [
        (10, vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8, 9]),
        (65, vec![0, 1, 31, 64], vec![2, 3, 30, 32, 62, 63]),
        (144, vec![0, 63, 64, 129], vec![1, 62, 65, 127, 128, 143]),
    ] {
        let mut state = state_for(&block, n, &source, &target);
        let outside = (0..n as usize)
            .filter(|i| !source.contains(i) && !target.contains(i))
            .collect::<Vec<_>>();
        if outside.len() >= 2 {
            state
                .apply_cnot(CnotGate {
                    control: outside[0] as u32,
                    target: outside[1] as u32,
                })
                .unwrap();
        }
        check(&[state], &source, &target, PaidShearConfig::source_exact());
    }
}

#[test]
fn gpu_paid_shear_preserves_group_order_ties_zero_sweeps_and_complete_small_blocks() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let source = [0, 2];
    let target = [1, 3, 4];
    let states = (0..64)
        .map(|bits| {
            let block = (0..3)
                .map(|r| (0..2).map(|c| ((bits >> (r * 2 + c)) & 1) as u8).collect())
                .collect::<Vec<_>>();
            state_for(&block, 5, &source, &target)
        })
        .collect::<Vec<_>>();
    for (local_width, sweeps) in [(2, 0), (2, 1), (3, 2), (4, 2), (4, 16)] {
        check(
            &states,
            &source,
            &target,
            PaidShearConfig {
                local_width,
                sweeps,
            },
        );
    }
    let source = [0, 2, 4];
    let target = [1, 3, 5];
    let states = (0..512)
        .map(|bits| {
            let block = (0..3)
                .map(|r| (0..3).map(|c| ((bits >> (r * 3 + c)) & 1) as u8).collect())
                .collect::<Vec<_>>();
            state_for(&block, 6, &source, &target)
        })
        .collect::<Vec<_>>();
    check(
        &states,
        &source,
        &target,
        PaidShearConfig {
            local_width: 3,
            sweeps: 2,
        },
    );
}

#[test]
fn gpu_paid_shear_preserves_both_sides_and_multiple_local_moves() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let source = (0..8).collect::<Vec<_>>();
    let target = (8..16).collect::<Vec<_>>();
    let mut seed = 0x81edb1e7a32f401du64;
    let states = (0..12)
        .map(|sample| {
            let block = (0..8)
                .map(|r| {
                    (0..8)
                        .map(|c| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            if sample == 0 {
                                u8::from(r < 7 && c < 7)
                            } else {
                                ((seed >> 11) & 1) as u8
                            }
                        })
                        .collect()
                })
                .collect::<Vec<_>>();
            state_for(&block, 16, &source, &target)
        })
        .collect::<Vec<_>>();
    let totals = check(
        &states,
        &source,
        &target,
        PaidShearConfig {
            local_width: 4,
            sweeps: 16,
        },
    );
    assert!(
        totals.0 > 1 && totals.1 > 1,
        "cover accepted moves on both sides: {totals:?}"
    );
}

#[test]
fn gpu_paid_shear_handles_vectors_wider_than_one_cta_and_explicit_alphabet_guards() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    for (source, target) in [
        ((0..129).collect::<Vec<_>>(), vec![130, 131, 143]),
        (vec![130, 131, 143], (0..129).collect::<Vec<_>>()),
    ] {
        let block = vec![vec![1; source.len()]; target.len()];
        let state = state_for(&block, 144, &source, &target);
        check(
            &[state],
            &source,
            &target,
            PaidShearConfig {
                local_width: 2,
                sweeps: 2,
            },
        );
    }
    let a = CnotAlphabet::all_to_all_unit(4).unwrap();
    let wrong = CnotAlphabet::new(
        4,
        "wrong-paid-shear",
        a.gates().to_vec(),
        vec![1; a.gates().len()],
    )
    .unwrap();
    let resident = tensor_states(&[MatrixState::identity(4, wrong.id()).unwrap()], &wrong);
    let plan = DevicePaidShearPlan::new(
        PaidBasisBlockContext::new(4, vec![0, 1], vec![2, 3]).unwrap(),
        PaidShearConfig::source_exact(),
        0,
    )
    .unwrap();
    let result = plan.compile(&resident).unwrap();
    assert_eq!(ints(result.status()), [3]);
    for tensor in [
        result.word(),
        result.lengths(),
        result.statistics(),
        result.endpoint(),
        result.transformed_block(),
        result.source_basis(),
        result.target_basis(),
        result.preparation(),
        result.restoration(),
        result.core(),
        result.accepted_moves(),
        result.cancellation_events(),
    ] {
        assert!(ints(tensor).iter().all(|&v| v == 0));
    }
    let raw =
        DeviceStateBatch::from_packed_tensor(resident.tensor().shallow_clone(), 4, 0).unwrap();
    assert!(plan.compile(&raw).err().unwrap().contains("alphabet"));
}
