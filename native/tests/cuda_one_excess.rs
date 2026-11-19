use cnot103_native::device_one_excess::DeviceOneExcessPlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::{recognize_stage38, CnotAlphabet, CnotGate, MatrixState, Stage38Result};
use num_bigint::{BigInt, Sign};
use tch::{Device, Kind, Tensor};

#[path = "cuda_one_excess/feature_plan.rs"]
mod feature_plan;

fn ints(t: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).flatten(0, -1)).unwrap()
}
fn resident(states: &[MatrixState], a: &CnotAlphabet) -> DeviceStateBatch {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let values: Vec<_> = states
        .iter()
        .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
        .collect();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&values)
            .view([
                states.len() as i64,
                a.dimension() as i64,
                a.dimension().div_ceil(64) as i64,
            ])
            .to_device(Device::Cuda(0)),
        a,
        0,
    )
    .unwrap()
}
fn all(n: u32) -> (CnotAlphabet, Vec<MatrixState>) {
    let a = CnotAlphabet::all_to_all_unit(n).unwrap();
    let states = (0u32..1 << (n * n))
        .filter_map(|bits| {
            MatrixState::from_packed_rows(
                n,
                a.id(),
                (0..n)
                    .map(|r| u64::from((bits >> (r * n)) & ((1 << n) - 1)))
                    .collect(),
            )
            .ok()
        })
        .collect();
    (a, states)
}
fn compare(states: &[MatrixState], a: &CnotAlphabet, budget: usize, max_n: u32) -> [usize; 4] {
    let device = resident(states, a);
    let output = DeviceOneExcessPlan::new(a.dimension(), 0, max_n, budget)
        .unwrap()
        .evaluate(&device)
        .unwrap();
    let headers = ints(output.headers());
    let lengths = ints(output.lengths());
    let gates = ints(output.encoder_gates());
    let status = ints(output.status());
    let features =
        Vec::<f32>::try_from(output.feature().to_device(Device::Cpu).flatten(0, -1)).unwrap();
    let words = ints(output.counts());
    let limbs = output.counts().size()[1] as usize;
    let capacity = a.dimension() as usize + 1;
    let mut classes = [0; 4];
    for (b, state) in states.iter().enumerate() {
        let h = &headers[b * 6..b * 6 + 6];
        let word = &gates[b * 2 * capacity..(b + 1) * 2 * capacity];
        let bytes: Vec<_> = words[b * limbs..(b + 1) * limbs]
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect();
        let count = BigInt::from_bytes_le(Sign::Plus, &bytes);
        let identity = MatrixState::identity(a.dimension(), a.id()).unwrap();
        let changed = (0..a.dimension())
            .filter(|&r| state.row_words(r) != identity.row_words(r))
            .count() as i64;
        if a.dimension() > max_n || a.id() != format!("all_to_all_unit_cnot:{}", a.dimension()) {
            assert_eq!(status[b], if a.dimension() > max_n { 4 } else { 3 });
            assert!(h.iter().chain(word).all(|&v| v == 0));
            assert_eq!(lengths[b], 0);
            assert_eq!(count, BigInt::from(0));
            assert_eq!(&features[b * 3..b * 3 + 3], &[0.0; 3]);
            continue;
        }
        let result = recognize_stage38(state, a, budget).unwrap();
        match result {
            Stage38Result::Exact(exact) => {
                classes[0] += 1;
                assert_eq!(status[b], 1);
                assert_eq!(
                    h,
                    &[
                        1,
                        changed,
                        exact.templates as i64,
                        exact.max_bag as i64,
                        i64::from(exact.transport_free),
                        exact.distance as i64
                    ]
                );
                assert_eq!(count, exact.one_excess_count.parse::<BigInt>().unwrap());
                assert_eq!(lengths[b], exact.distance as i64);
                let expected: Vec<_> = exact
                    .word
                    .gates
                    .iter()
                    .rev()
                    .flat_map(|g| [g.control as i64, g.target as i64])
                    .collect();
                assert_eq!(&word[..expected.len()], expected);
                assert!(word[expected.len()..].iter().all(|&v| v == 0));
                let mut replay = identity;
                for gate in word[..expected.len()].chunks_exact(2) {
                    replay
                        .apply_cnot(CnotGate {
                            control: gate[0] as u32,
                            target: gate[1] as u32,
                        })
                        .unwrap();
                }
                assert_eq!(replay, *state);
                assert_eq!(&features[b * 3..b * 3 + 3], &[1.0, 0.0, 0.0]);
            }
            Stage38Result::CertifiedNoSolutionWithinScope(no) => {
                classes[1] += 1;
                assert_eq!(status[b], 1);
                assert_eq!(
                    h,
                    &[
                        2,
                        changed,
                        no.templates as i64,
                        no.max_bag as i64,
                        0,
                        no.excluded_length as i64
                    ]
                );
                assert_eq!(count, BigInt::from(0));
                assert_eq!(&features[b * 3..b * 3 + 3], &[0.0, 1.0, 0.0]);
                assert_eq!(lengths[b], 0);
                assert!(word.iter().all(|&v| v == 0));
            }
            Stage38Result::Unsupported(_) => {
                classes[2] += 1;
                assert_eq!(status[b], 1);
                assert_eq!(h, &[3, changed, 0, 0, 0, 0]);
                assert_eq!(count, BigInt::from(0));
                assert_eq!(&features[b * 3..b * 3 + 3], &[0.0, 0.0, 1.0]);
                assert_eq!(lengths[b], 0);
                assert!(word.iter().all(|&v| v == 0));
            }
            Stage38Result::ResourceLimit(_) => {
                classes[3] += 1;
                assert_eq!(status[b], 4);
                assert_eq!(h, &[4, changed, 0, 0, 0, 0]);
                assert_eq!(count, BigInt::from(0));
                assert_eq!(&features[b * 3..b * 3 + 3], &[0.0; 3]);
                assert_eq!(lengths[b], 0);
                assert!(word.iter().all(|&v| v == 0));
            }
        }
    }
    assert_eq!(
        ints(device.tensor()),
        states
            .iter()
            .flat_map(|s| s.packed_rows().iter().map(|&v| v as i64))
            .collect::<Vec<_>>()
    );
    classes
}
#[test]
fn gpu_one_excess_matches_complete_gl2_gl3_results_and_words() {
    for n in [2, 3] {
        let (a, states) = all(n);
        assert_eq!(states.len(), if n == 2 { 6 } else { 168 });
        for budget in [1, 8, 16, 64, 100_000] {
            let classes = compare(&states, &a, budget, n);
            if n == 3 && budget == 100_000 {
                assert_eq!(classes, [75, 35, 58, 0]);
            }
        }
    }
}
#[test]
fn gpu_one_excess_preserves_full_packed_physical_order() {
    let (_, small) = all(3);
    for n in [5, 65, 144] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let wires = [n - 1, 0, n / 2];
        let mut cases = Vec::new();
        for (i, s) in small.iter().enumerate() {
            if i % 7 != 0 {
                continue;
            }
            let mut rows = MatrixState::identity(n, a.id())
                .unwrap()
                .packed_rows()
                .to_vec();
            let w = n.div_ceil(64) as usize;
            for (r, &physical) in wires.iter().enumerate() {
                rows[physical as usize * w..(physical as usize + 1) * w].fill(0);
                for (c, &column) in wires.iter().enumerate() {
                    if s.packed_rows()[r] >> c & 1 != 0 {
                        rows[physical as usize * w + column as usize / 64] |= 1 << (column % 64);
                    }
                }
            }
            cases.push(MatrixState::from_packed_rows(n, a.id(), rows).unwrap());
        }
        for budget in [8, 100_000] {
            compare(&cases, &a, budget, n);
        }
    }
    let (a, all4) = all(4);
    let cases: Vec<_> = all4
        .into_iter()
        .enumerate()
        .filter_map(|(i, s)| (i % 503 == 0).then_some(s))
        .collect();
    for budget in [16, 100_000] {
        compare(&cases, &a, budget, 4);
    }
}
#[test]
fn gpu_one_excess_preserves_width_and_alphabet_priority() {
    let a = CnotAlphabet::all_to_all_unit(3).unwrap();
    let identity = MatrixState::identity(3, a.id()).unwrap();
    compare(&[identity.clone()], &a, 17, 2);
    let custom = CnotAlphabet::new(
        3,
        "one-excess-custom",
        vec![CnotGate {
            control: 0,
            target: 1,
        }],
        vec![2],
    )
    .unwrap();
    let state = MatrixState::identity(3, custom.id()).unwrap();
    compare(&[state.clone()], &custom, 17, 2);
    compare(&[state], &custom, 17, 3);
    assert!(DeviceOneExcessPlan::new(3, 0, 3, 0).is_err());
    assert!(DeviceOneExcessPlan::new(1, 0, 3, 17).is_err());
    assert!(DeviceOneExcessPlan::new(3, 0, 1, 17).is_err());
    let gpu = resident(&[identity], &a);
    let unlabeled =
        DeviceStateBatch::from_packed_tensor(gpu.tensor().shallow_clone(), 3, 0).unwrap();
    assert!(DeviceOneExcessPlan::new(3, 0, 3, 17)
        .unwrap()
        .evaluate(&unlabeled)
        .is_err());
}

#[test]
fn gpu_one_excess_preserves_large_total_and_complete_recovery() {
    for n in [65, 144] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let mut state = MatrixState::identity(n, a.id()).unwrap();
        for (control, target) in [(1, 0), (0, 1), (1, 0)] {
            state.apply_cnot(CnotGate { control, target }).unwrap();
        }
        let mut separate = 0;
        for control in (2..n - 1).step_by(2) {
            state
                .apply_cnot(CnotGate {
                    control,
                    target: control + 1,
                })
                .unwrap();
            separate += 1;
        }
        let expected: BigInt = (2..=separate + 3).fold(BigInt::from(1), |a, b| a * b) / 3;
        let exact = recognize_stage38(&state, &a, 27).unwrap();
        let Stage38Result::Exact(exact) = exact else {
            panic!("independent swap/shear family was not exact")
        };
        assert_eq!(exact.one_excess_count, expected.to_string());
        assert_eq!(exact.templates, 2);
        assert_eq!(exact.max_bag, 3);
        if n == 144 {
            assert!(expected.bits() > 128);
        }
        compare(&[state.clone()], &a, 26, n);
        compare(&[state], &a, 27, n);
    }
}

#[test]
fn gpu_one_excess_resource_overrides_prior_positive_results() {
    for (n, rows, budget) in [
        (3, vec![3, 5, 1], 16),
        (8, vec![17, 130, 128, 40, 88, 160, 104, 132], 729),
    ] {
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let state = MatrixState::from_packed_rows(n, a.id(), rows).unwrap();
        assert_eq!(compare(&[state.clone()], &a, budget, n), [0, 0, 0, 1]);
        assert_eq!(compare(&[state], &a, 100_000, n), [1, 0, 0, 0]);
    }
}
