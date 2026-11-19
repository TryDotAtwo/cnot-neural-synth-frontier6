//! Actual source-factor/order parity. CPU computation and readback occur only here.
use super::{factorization, ordering, tables};
use crate::stage38::Histories;
use std::collections::HashSet;
use std::ffi::c_void;
use tch::{Device, Kind, Tensor};

#[repr(C)]
struct Args {
    histories: *const u64,
    lengths: *const u32,
    protected: *const u8,
    output: *mut u32,
    bitsets: *mut u64,
    scratch: *mut u64,
    stream: *mut c_void,
    history_words: u64,
    length_elements: u64,
    protected_elements: u64,
    output_elements: u64,
    bitset_words: u64,
    scratch_words: u64,
    max_entries: u64,
    batch: u32,
    dimension: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_one_excess_structure(args: *const Args) -> i32;
}

struct Layout {
    n: usize,
    w: usize,
    k: usize,
    f: usize,
    t: usize,
    ew: usize,
    term: usize,
    ids: usize,
    vertices: usize,
    offsets: usize,
    order: usize,
    bag: usize,
    budget: usize,
    protected: usize,
    meta: usize,
    bits: usize,
    scratch: usize,
}
impl Layout {
    fn new(n: usize) -> Self {
        let (w, k, f, t) = (n.div_ceil(64), n + 1, n + 2, 3 * n);
        let ew = k.div_ceil(64);
        let term = 8 + 3 * k + f;
        let ids = term + 5 * t * f;
        let vertices = ids + k;
        let offsets = vertices + k;
        let order = offsets + k + 1;
        let bag = order + k;
        let budget = bag + k;
        let protected = budget + k;
        Self {
            n,
            w,
            k,
            f,
            t,
            ew,
            term,
            ids,
            vertices,
            offsets,
            order,
            bag,
            budget,
            protected,
            meta: protected + k,
            bits: (f + 2 * k) * ew,
            scratch: n * w + k * ew + 7 * k + 16,
        }
    }
    fn pair(&self, f: usize, t: usize, field: usize) -> usize {
        self.term + (field * self.t + t) * self.f + f
    }
}

fn expected(h: &Histories, retained: &HashSet<usize>, budget: usize) -> (Vec<i32>, Vec<i64>) {
    let l = Layout::new(h.len());
    let (events, factors) = factorization::history_factors(h).unwrap();
    let mut out = vec![0i32; l.meta];
    let mut bits = vec![0u64; l.bits];
    out[l.ids..l.ids + l.k].fill(l.k as i32);
    out[l.vertices..l.vertices + l.k].fill(l.k as i32);
    out[l.order..l.order + l.k].fill(l.k as i32);
    out[0] = events.len() as i32;
    out[1] = factors.len() as i32;
    out[4] = h.iter().position(|v| v.len() == 3).unwrap_or(l.n) as i32;
    out[6] = 1;
    let retained: HashSet<_> = retained
        .iter()
        .copied()
        .chain(
            h.iter()
                .enumerate()
                .filter_map(|(r, v)| (v.len() == 3).then_some(r)),
        )
        .collect();
    let protected: HashSet<_> = events
        .iter()
        .enumerate()
        .filter_map(|(e, (r, _))| retained.contains(r).then_some(e))
        .collect();
    out[5] = protected.len() as i32;
    for (e, &(r, t)) in events.iter().enumerate() {
        out[8 + e] = r as i32;
        out[8 + l.k + e] = t as i32;
    }
    let mut count = 0;
    for (r, row) in h.iter().enumerate() {
        out[8 + 2 * l.k + r] = count;
        count += (row.len() - 1) as i32;
    }
    out[8 + 2 * l.k + l.n] = count;
    for (f, factor) in factors.iter().enumerate() {
        out[8 + 3 * l.k + f] = factor.terms.len() as i32;
        for &v in &factor.scope {
            bits[(v / 64) * l.f + f] |= 1u64 << (v % 64);
        }
        for (t, term) in factor.terms.iter().enumerate() {
            out[l.pair(f, t, 0)] = term.len() as i32;
            for (p, &(a, b)) in term.iter().enumerate() {
                out[l.pair(f, t, 1 + 2 * p)] = a as i32;
                out[l.pair(f, t, 2 + 2 * p)] = b as i32;
            }
        }
    }
    if factors.iter().any(|f| f.terms.is_empty()) {
        out[2] = 1;
        return (out, bits.into_iter().map(|x| x as i64).collect());
    }
    for factor in &factors {
        for &a in &factor.scope {
            for &b in &factor.scope {
                if a != b {
                    bits[l.f * l.ew + (b / 64) * l.k + a] |= 1u64 << (b % 64);
                }
            }
        }
        for &(a, b) in &factor.terms[0] {
            if factor.terms.iter().all(|term| term.contains(&(a, b))) {
                bits[(l.f + l.k) * l.ew + (b / 64) * l.k + a] |= 1u64 << (b % 64);
            }
        }
    }
    if ordering::has_mandatory_cycle(events.len(), &factors) {
        out[2] = 2;
        return (out, bits.into_iter().map(|x| x as i64).collect());
    }
    let active: Vec<_> = factors
        .into_iter()
        .filter(|f| !f.scope.is_empty())
        .collect();
    let components = ordering::components(events.len(), &active);
    out[3] = components.len() as i32;
    let mut offset = 0;
    for (c, vertices) in components.iter().enumerate() {
        let set: HashSet<_> = vertices.iter().copied().collect();
        let fs: Vec<_> = active
            .iter()
            .filter(|f| f.scope.iter().all(|v| set.contains(v)))
            .cloned()
            .collect();
        let (order, bag) = ordering::elimination_order(vertices, &fs, &protected).unwrap();
        out[l.offsets + c] = offset as i32;
        for (&v, &o) in vertices.iter().zip(&order) {
            out[l.ids + v] = c as i32;
            out[l.vertices + offset] = v as i32;
            out[l.order + offset] = o as i32;
            offset += 1;
        }
        out[l.offsets + c + 1] = offset as i32;
        out[l.bag + c] = bag as i32;
        out[l.budget + c] = tables::checked_power(vertices.len(), bag, budget).is_ok() as i32;
        // Independent edge/vertex count for the unprotected induced subgraph.
        let mut seen = HashSet::new();
        let mut pseudo = true;
        for &start in vertices {
            if protected.contains(&start) || seen.contains(&start) {
                continue;
            }
            let mut queue = vec![start];
            seen.insert(start);
            let mut degree = 0;
            for head in 0..vertices.len() {
                if head >= queue.len() {
                    break;
                }
                let v = queue[head];
                for &u in vertices {
                    if !protected.contains(&u)
                        && bits[l.f * l.ew + (u / 64) * l.k + v] >> (u % 64) & 1 != 0
                    {
                        degree += 1;
                        if seen.insert(u) {
                            queue.push(u);
                        }
                    }
                }
            }
            if degree / 2 > queue.len() {
                pseudo = false;
            }
        }
        out[l.protected + c] = (!protected.is_empty() && pseudo) as i32;
    }
    (out, bits.into_iter().map(|x| x as i64).collect())
}

fn verify(cases: &[(Histories, HashSet<usize>)], budget: usize) {
    if cases.is_empty() {
        return;
    }
    let l = Layout::new(cases[0].0.len());
    let b = cases.len();
    let mut h = vec![0i64; b * 3 * l.w * l.n];
    let mut lens = vec![0i32; b * l.n];
    let mut p = vec![0u8; b * l.n];
    let mut expected_meta = Vec::new();
    let mut expected_bits = Vec::new();
    for (i, (rows, protected)) in cases.iter().enumerate() {
        for (r, row) in rows.iter().enumerate() {
            lens[i * l.n + r] = row.len() as i32;
            p[i * l.n + r] = protected.contains(&r) as u8;
            for (s, value) in row.iter().enumerate() {
                for (w, &v) in value.iter().enumerate() {
                    h[i * 3 * l.w * l.n + (s * l.w + w) * l.n + r] = v as i64;
                }
            }
        }
        let (m, w) = expected(rows, protected, budget);
        expected_meta.extend(m);
        expected_bits.extend(w);
    }
    let dev = Device::Cuda(0);
    let ht = Tensor::from_slice(&h).to_device(dev);
    let lt = Tensor::from_slice(&lens).to_device(dev);
    let pt = Tensor::from_slice(&p).to_device(dev);
    let out = Tensor::empty([(b * l.meta) as i64], (Kind::Int, dev));
    let bits = Tensor::empty([(b * l.bits) as i64], (Kind::Int64, dev));
    let scratch = Tensor::empty([(b * l.scratch) as i64], (Kind::Int64, dev));
    let args = Args {
        histories: ht.data_ptr().cast(),
        lengths: lt.data_ptr().cast(),
        protected: pt.data_ptr().cast(),
        output: out.data_ptr().cast(),
        bitsets: bits.data_ptr().cast(),
        scratch: scratch.data_ptr().cast(),
        stream: crate::cuda::current_stream(0).unwrap(),
        history_words: h.len() as u64,
        length_elements: lens.len() as u64,
        protected_elements: p.len() as u64,
        output_elements: out.numel() as u64,
        bitset_words: bits.numel() as u64,
        scratch_words: scratch.numel() as u64,
        max_entries: budget as u64,
        batch: b as u32,
        dimension: l.n as u32,
    };
    assert_eq!(
        unsafe { cnot103_cuda_launch_one_excess_structure(&args) },
        0
    );
    let got = Vec::<i32>::try_from(out.to_device(Device::Cpu)).unwrap();
    for (index, (&a, &e)) in got.iter().zip(&expected_meta).enumerate() {
        assert_eq!(
            a,
            e,
            "n={} budget={budget} case={} meta={}",
            l.n,
            index / l.meta,
            index % l.meta
        );
    }
    assert_eq!(
        Vec::<i64>::try_from(bits.to_device(Device::Cpu)).unwrap(),
        expected_bits,
        "n={} budget={budget} bitsets",
        l.n
    );
    assert_eq!(Vec::<i64>::try_from(ht.to_device(Device::Cpu)).unwrap(), h);
    assert_eq!(
        Vec::<i32>::try_from(lt.to_device(Device::Cpu)).unwrap(),
        lens
    );
    assert_eq!(Vec::<u8>::try_from(pt.to_device(Device::Cpu)).unwrap(), p);
}

fn unit(n: usize, r: usize) -> Vec<u64> {
    let mut v = vec![0; n.div_ceil(64)];
    v[r / 64] = 1 << (r % 64);
    v
}
fn from_word(n: usize, word: &[(usize, usize)]) -> Histories {
    let mut h: Histories = (0..n).map(|r| vec![unit(n, r)]).collect();
    for &(c, t) in word {
        let next = h[c]
            .last()
            .unwrap()
            .iter()
            .zip(h[t].last().unwrap())
            .map(|(a, b)| a ^ b)
            .collect();
        h[t].push(next);
    }
    h
}
#[test]
fn cuda_one_excess_structure_matches_all_gl3_histories_and_budgets() {
    crate::torch_device::require_cuda_device(0).unwrap();
    let mut cases = Vec::new();
    for code in 0..512u64 {
        let rows: Vec<_> = (0..3).map(|r| (code >> (3 * r)) & 7).collect();
        if crate::state::MatrixState::from_packed_rows(3, "test", rows.clone()).is_err() {
            continue;
        }
        for target in 0..3 {
            if rows[target] == 1 << target {
                continue;
            }
            for middle in 1..8u64 {
                if middle == rows[target] || middle == 1 << target {
                    continue;
                }
                let h: Histories = (0..3)
                    .map(|r| {
                        if r == target {
                            vec![vec![rows[r]], vec![middle], vec![1 << r]]
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
    assert!(cases.len() > 2000);
    for budget in [1, 2, 4, 16, 27, 64, 125, 243, 1024, usize::MAX] {
        for chunk in cases.chunks(128) {
            verify(chunk, budget);
        }
    }
}
#[test]
fn cuda_one_excess_structure_preserves_physical_history_and_protected_recovery() {
    for n in [2, 5, 8, 65, 144] {
        let mut cases = vec![(from_word(n, &[]), HashSet::new())];
        let word: Vec<_> = (0..n - 1)
            .map(|t| (t + 1, t))
            .chain(std::iter::once((n - 1, 0)))
            .collect();
        let h = from_word(n, &word);
        cases.push((h.clone(), HashSet::new()));
        let mut recovery = h.clone();
        recovery[0].remove(0);
        cases.push((recovery, HashSet::from([0])));
        if n >= 3 {
            let wires = [n - 1, 0, n / 2];
            let w = [
                (wires[1], wires[0]),
                (wires[2], wires[1]),
                (wires[1], wires[0]),
            ];
            let h = from_word(n, &w);
            cases.push((h.clone(), HashSet::from([n - 1])));
            let mut r = h;
            r[n - 1].remove(0);
            cases.push((r, HashSet::from([n - 1])));
        }
        for budget in [1, 4, 16, 64, 1024, usize::MAX] {
            verify(&cases, budget);
        }
    }
    let mut seed = 0xF08A615u64;
    for n in [5, 8] {
        let mut cases = Vec::new();
        for _ in 0..192 {
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as usize
            };
            let double = next() % n;
            let mut targets: Vec<_> = (0..n).chain(std::iter::once(double)).collect();
            for i in (1..targets.len()).rev() {
                targets.swap(i, next() % (i + 1));
            }
            let word: Vec<_> = targets
                .into_iter()
                .map(|t| {
                    let c = (t + 1 + next() % (n - 1)) % n;
                    (c, t)
                })
                .collect();
            cases.push((from_word(n, &word), HashSet::from([double, next() % n])));
        }
        for budget in [1, 64, 1024, usize::MAX] {
            for chunk in cases.chunks(64) {
                verify(chunk, budget);
            }
        }
    }
}
