//! Source BigInt parity, including eviction/recomputation and wide counts.
use super::count::count_histories;
use crate::stage38::{Histories, Stage38Error};
use num_bigint::{BigInt, Sign};
use std::collections::HashSet;
use std::ffi::c_void;
use tch::{Device, Kind, Tensor};

#[repr(C)]
struct Args {
    histories: *const u64,
    lengths: *const u32,
    protected_rows: *const u8,
    structure_meta: *mut u32,
    structure_bits: *mut u64,
    structure_scratch: *mut u64,
    program: *mut u64,
    arithmetic: *mut u32,
    counts: *mut u32,
    results: *mut u64,
    stream: *mut c_void,
    history_words: u64,
    length_elements: u64,
    protected_elements: u64,
    meta_elements: u64,
    bit_words: u64,
    scratch_words: u64,
    program_words: u64,
    arithmetic_elements: u64,
    count_elements: u64,
    result_words: u64,
    max_entries: u64,
    batch: u32,
    dimension: u32,
    workers: u32,
    lanes: u32,
    cache_slots: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_one_excess_count(args: *const Args) -> i32;
}
fn capacity(n: usize) -> usize {
    let k = n + 1;
    let bits = 1
        + 2 * n.ilog2() as usize
        + 2
        + k
        + 2 * k * (k.ilog2() as usize + 1)
        + (k + 1) * ((3 * n).ilog2() as usize + 1);
    bits.div_ceil(32) + 1
}
fn verify(
    cases: &[(Histories, HashSet<usize>)],
    budget: usize,
    cache: usize,
    lanes: usize,
    workers: usize,
) -> Vec<Vec<u64>> {
    crate::torch_device::require_cuda_device(0).unwrap();
    let n = cases[0].0.len();
    let b = cases.len();
    let workers = workers.min(b);
    let w = n.div_ceil(64);
    let k = n + 1;
    let f = n + 2;
    let ew = k.div_ceil(64);
    let limbs = capacity(n);
    let nodes = f + k;
    let depth = k + 1;
    let meta = 9 + 10 * k + f + 15 * n * f;
    let bits = (f + 2 * k) * ew;
    let scratch = n * w + k * ew + 7 * k + 16;
    let program =
        16 + 8 * nodes + nodes * ew + k * lanes + 2 * cache * lanes + 5 * depth * lanes + 2 * lanes;
    let arithmetic = limbs * ((2 * depth + 3 + cache) * lanes + k + 1 + 5);
    let mut h = vec![0i64; b * 3 * w * n];
    let mut lengths = vec![0i32; b * n];
    let mut protected = vec![0u8; b * n];
    for (i, (rows, retained)) in cases.iter().enumerate() {
        for (r, row) in rows.iter().enumerate() {
            lengths[i * n + r] = row.len() as i32;
            protected[i * n + r] = retained.contains(&r) as u8;
            for (p, v) in row.iter().enumerate() {
                for (word, &value) in v.iter().enumerate() {
                    h[i * 3 * w * n + (p * w + word) * n + r] = value as i64;
                }
            }
        }
    }
    let device = Device::Cuda(0);
    let ht = Tensor::from_slice(&h).to_device(device);
    let lt = Tensor::from_slice(&lengths).to_device(device);
    let pt = Tensor::from_slice(&protected).to_device(device);
    let allocate = |size: usize, kind| Tensor::empty([size as i64], (kind, device));
    let sm = allocate(workers * meta, Kind::Int);
    let sb = allocate(workers * bits, Kind::Int64);
    let ss = allocate(workers * scratch, Kind::Int64);
    let pg = allocate(workers * program, Kind::Int64);
    let ar = allocate(workers * arithmetic, Kind::Int);
    let counts = allocate(b * limbs, Kind::Int);
    let results = allocate(b * 4, Kind::Int64);
    let args = Args {
        histories: ht.data_ptr().cast(),
        lengths: lt.data_ptr().cast(),
        protected_rows: pt.data_ptr().cast(),
        structure_meta: sm.data_ptr().cast(),
        structure_bits: sb.data_ptr().cast(),
        structure_scratch: ss.data_ptr().cast(),
        program: pg.data_ptr().cast(),
        arithmetic: ar.data_ptr().cast(),
        counts: counts.data_ptr().cast(),
        results: results.data_ptr().cast(),
        stream: crate::cuda::current_stream(0).unwrap(),
        history_words: h.len() as u64,
        length_elements: lengths.len() as u64,
        protected_elements: protected.len() as u64,
        meta_elements: sm.numel() as u64,
        bit_words: sb.numel() as u64,
        scratch_words: ss.numel() as u64,
        program_words: pg.numel() as u64,
        arithmetic_elements: ar.numel() as u64,
        count_elements: counts.numel() as u64,
        result_words: results.numel() as u64,
        max_entries: budget as u64,
        batch: b as u32,
        dimension: n as u32,
        workers: workers as u32,
        lanes: lanes as u32,
        cache_slots: cache as u32,
    };
    assert_eq!(unsafe { cnot103_cuda_launch_one_excess_count(&args) }, 0);
    let words = Vec::<i32>::try_from(counts.to_device(Device::Cpu)).unwrap();
    let headers = Vec::<i64>::try_from(results.to_device(Device::Cpu)).unwrap();
    for (i, (rows, retained)) in cases.iter().enumerate() {
        let bytes: Vec<u8> = words[i * limbs..(i + 1) * limbs]
            .iter()
            .flat_map(|v| (*v as u32).to_le_bytes())
            .collect();
        let got = BigInt::from_bytes_le(Sign::Plus, &bytes);
        match count_histories(rows, budget, retained) {
            Ok(audit) => {
                assert_eq!(
                    headers[4 * i],
                    1,
                    "n{n} case{i} budget{budget} cache{cache}"
                );
                assert_eq!(
                    headers[4 * i + 1],
                    audit.max_bag as i64,
                    "max_bag n{n} case{i}"
                );
                assert_eq!(
                    got, audit.count,
                    "count n{n} case{i} budget{budget} cache{cache} lanes{lanes}"
                );
            }
            Err(Stage38Error::ContractionBudgetExceeded(_)) => {
                assert_eq!(headers[4 * i], 4, "resource n{n} case{i} budget{budget}");
                assert_eq!(headers[4 * i + 1], 0);
                assert_eq!(got, BigInt::from(0));
            }
            Err(other) => panic!("unexpected source outcome {other:?}"),
        }
        if cache == 0 {
            assert_eq!(headers[4 * i + 3], 0);
        }
    }
    assert_eq!(Vec::<i64>::try_from(ht.to_device(Device::Cpu)).unwrap(), h);
    assert_eq!(
        Vec::<i32>::try_from(lt.to_device(Device::Cpu)).unwrap(),
        lengths
    );
    assert_eq!(
        Vec::<u8>::try_from(pt.to_device(Device::Cpu)).unwrap(),
        protected
    );
    headers
        .chunks(4)
        .map(|h| h.iter().map(|v| *v as u64).collect())
        .collect()
}
fn unit(n: usize, r: usize) -> Vec<u64> {
    let mut row = vec![0; n.div_ceil(64)];
    row[r / 64] = 1 << (r % 64);
    row
}
fn from_word(n: usize, word: &[(usize, usize)]) -> Histories {
    let mut h: Histories = (0..n).map(|r| vec![unit(n, r)]).collect();
    for &(c, t) in word {
        let v = h[c]
            .last()
            .unwrap()
            .iter()
            .zip(h[t].last().unwrap())
            .map(|(a, b)| a ^ b)
            .collect();
        h[t].push(v);
    }
    h
}
fn small_histories() -> Vec<(Histories, HashSet<usize>)> {
    let mut cases = Vec::new();
    for code in 0..512u64 {
        let rows: Vec<_> = (0..3).map(|r| (code >> (3 * r)) & 7).collect();
        if crate::state::MatrixState::from_packed_rows(3, "test", rows.clone()).is_err() {
            continue;
        }
        for t in 0..3 {
            if rows[t] == 1 << t {
                continue;
            }
            for mid in 1..8u64 {
                if mid == rows[t] || mid == 1 << t {
                    continue;
                }
                let h = (0..3)
                    .map(|r| {
                        if r == t {
                            vec![vec![rows[r]], vec![mid], vec![1 << r]]
                        } else if rows[r] == 1 << r {
                            vec![vec![rows[r]]]
                        } else {
                            vec![vec![rows[r]], vec![1 << r]]
                        }
                    })
                    .collect();
                cases.push((h, HashSet::new()));
            }
        }
    }
    cases
}
#[test]
fn cuda_one_excess_count_matches_all_gl3_counts_with_eviction_and_budgets() {
    let cases = small_histories();
    assert_eq!(cases.len(), 2160);
    let mut evictions = 0;
    for budget in [1, 16, 64, usize::MAX] {
        for (cache, lanes, workers) in [(0, 1, 1), (1, 2, 3), (127, 4, 4)] {
            for chunk in cases.chunks(128) {
                for header in verify(chunk, budget, cache, lanes, workers) {
                    evictions += header[3];
                }
            }
        }
    }
    assert!(evictions > 0, "forced recomputation was not exercised");
}
#[test]
fn cuda_one_excess_count_preserves_wide_integers_recovery_and_late_resource() {
    for n in [2, 5, 8, 65, 144] {
        let mut word = Vec::new();
        if n >= 3 {
            word.extend([(1, 0), (2, 0)]);
            for t in (3..n - 1).step_by(2) {
                word.push((t + 1, t));
            }
        }
        let h = from_word(n, &word);
        let mut cases = vec![
            (from_word(n, &[]), HashSet::new()),
            (h.clone(), HashSet::new()),
        ];
        if n >= 3 {
            let mut r = h;
            r[0].remove(0);
            cases.push((r, HashSet::from([0])));
        }
        for budget in [1, 4, 64, usize::MAX] {
            verify(&cases, budget, 1, 1, 1);
            verify(&cases, budget, 257, 4, 2);
        }
        if n == 144 {
            assert!(
                count_histories(&cases[1].0, usize::MAX, &HashSet::new())
                    .unwrap()
                    .count
                    .bits()
                    > 128
            );
        }
    }
    let h = from_word(8, &[(1, 0), (3, 2), (4, 3), (5, 4), (6, 5)]);
    let result = verify(&[(h, HashSet::new())], 8, 17, 3, 1);
    assert_eq!(result[0][0], 4);
    assert_eq!(
        result[0][2], 2,
        "budget failed before the positive first component"
    );
}
#[test]
fn cuda_one_excess_count_matches_connected_histories() {
    let mut seed = 0xF08C0A17u64;
    for n in [5, 8] {
        let mut cases = Vec::new();
        for _ in 0..24 {
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as usize
            };
            let twice = next() % n;
            let mut targets: Vec<_> = (0..n).chain(std::iter::once(twice)).collect();
            for i in (1..targets.len()).rev() {
                targets.swap(i, next() % (i + 1));
            }
            let word: Vec<_> = targets
                .into_iter()
                .map(|t| ((t + 1 + next() % (n - 1)) % n, t))
                .collect();
            cases.push((from_word(n, &word), HashSet::from([twice])));
        }
        for budget in [1, 64, 4096] {
            verify(&cases, budget, 8192, 4, 2);
        }
    }
}

#[test]
fn cuda_one_excess_count_cancels_wide_connected_finite_differences() {
    // All 38 leaf updates precede the single center update: 38! legal words.
    // Unlike disconnected factorial fixtures, the lattice values and signed
    // finite-difference prefixes themselves require more than 128 bits.
    let n = 40;
    let mut word: Vec<_> = (1..n - 1).map(|t| (0, t)).collect();
    word.push((n - 1, 0));
    let cases = [(from_word(n, &word), HashSet::from([0]))];
    let expected = (2..n - 1).fold(BigInt::from(1), |v, f| v * f);
    assert!(expected.bits() > 128);
    let source = count_histories(&cases[0].0, usize::MAX, &cases[0].1).unwrap();
    assert_eq!(source.count, expected);
    assert_eq!(source.max_bag, 2);
    for budget in [1520, 1521, usize::MAX] {
        for (cache, lanes) in [(1, 1), (4096, 8)] {
            let headers = verify(&cases, budget, cache, lanes, 1);
            assert_eq!(headers[0][2], 1);
        }
    }
}
