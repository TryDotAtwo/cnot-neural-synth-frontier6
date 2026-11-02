//! NumPy `default_rng(u64)` compatibility for source-defined Monte Carlo inputs.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const PCG_MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

/// A small, source-versioned port of NumPy 2.2.6's `default_rng(u64)` path.
///
/// It implements SeedSequence, PCG64 XSL-RR 128/64, NumPy's bounded
/// `integers` sampler, and the Poisson algorithms used by `Generator.poisson`.
/// The feature producer version must include this compatibility identity since
/// `Generator` distribution streams do not carry NumPy's long-term guarantee.
#[derive(Clone, Debug)]
pub(crate) struct NumpyRng {
    state: u128,
    increment: u128,
    has_uint32: bool,
    cached_uint32: u32,
}

impl NumpyRng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut entropy = vec![seed as u32];
        let high = (seed >> 32) as u32;
        if high != 0 {
            entropy.push(high);
        }
        let pool = seed_sequence_pool(&entropy);
        let mut hash_const = INIT_B;
        let mut words = [0u32; 8];
        for (index, word) in words.iter_mut().enumerate() {
            let value = pool[index % pool.len()] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            let mixed = value.wrapping_mul(hash_const);
            *word = mixed ^ (mixed >> 16);
        }

        let initstate_high = u64::from(words[0]) | (u64::from(words[1]) << 32);
        let initstate_low = u64::from(words[2]) | (u64::from(words[3]) << 32);
        let initseq_high = u64::from(words[4]) | (u64::from(words[5]) << 32);
        let initseq_low = u64::from(words[6]) | (u64::from(words[7]) << 32);
        let state = (u128::from(initstate_high) << 64) | u128::from(initstate_low);
        let sequence = (u128::from(initseq_high) << 64) | u128::from(initseq_low);
        let increment = sequence.wrapping_shl(1) | 1;

        let mut rng = Self {
            state: 0,
            increment,
            has_uint32: false,
            cached_uint32: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(state);
        rng.step();
        rng
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step();
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((high >> 58) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.cached_uint32;
        }
        let value = self.next_u64();
        self.cached_uint32 = (value >> 32) as u32;
        self.has_uint32 = true;
        value as u32
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    pub(crate) fn integer(&mut self, upper_exclusive: usize) -> Option<usize> {
        let bound = u64::try_from(upper_exclusive).ok()?;
        if bound == 0 {
            return None;
        }
        if bound == 1 {
            return Some(0);
        }
        if bound <= u64::from(u32::MAX) + 1 {
            if bound == u64::from(u32::MAX) + 1 {
                return Some(self.next_u32() as usize);
            }
            let bound = bound as u32;
            let threshold = bound.wrapping_neg() % bound;
            loop {
                let product = u64::from(self.next_u32()) * u64::from(bound);
                let low = product as u32;
                if low < bound && low < threshold {
                    continue;
                }
                return usize::try_from((product >> 32) as u32).ok();
            }
        }

        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u128::from(self.next_u64()) * u128::from(bound);
            let low = product as u64;
            if low < bound && low < threshold {
                continue;
            }
            return usize::try_from((product >> 64) as u64).ok();
        }
    }

    pub(crate) fn poisson(&mut self, lambda: f64) -> Option<u64> {
        if !lambda.is_finite() || lambda < 0.0 || lambda > u64::MAX as f64 {
            return None;
        }
        if lambda >= 10.0 {
            return self.poisson_ptrs(lambda);
        }
        if lambda == 0.0 {
            return Some(0);
        }

        let threshold = (-lambda).exp();
        let mut product = 1.0;
        let mut count = 0u64;
        loop {
            product *= self.next_f64();
            if product > threshold {
                count = count.checked_add(1)?;
            } else {
                return Some(count);
            }
        }
    }

    fn poisson_ptrs(&mut self, lambda: f64) -> Option<u64> {
        let root = lambda.sqrt();
        let log_lambda = lambda.ln();
        let b = 0.931 + 2.53 * root;
        let a = -0.059 + 0.02483 * b;
        let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let quick_accept = 0.9277 - 3.6224 / (b - 2.0);

        loop {
            let u = self.next_f64() - 0.5;
            let v = self.next_f64();
            let us = 0.5 - u.abs();
            let k_float = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
            if (us >= 0.07) && (v <= quick_accept) {
                return u64::try_from(k_float as i64).ok();
            }
            if k_float < 0.0 || ((us < 0.013) && (v > us)) {
                continue;
            }
            let left = v.ln() + inverse_alpha.ln() - (a / (us * us) + b).ln();
            let right = -lambda + k_float * log_lambda - random_log_gamma(k_float + 1.0);
            if left <= right {
                return u64::try_from(k_float as i64).ok();
            }
        }
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
    }
}

fn seed_sequence_pool(entropy: &[u32]) -> [u32; 4] {
    let mut pool = [0u32; 4];
    let mut hash_const = INIT_A;
    for index in 0..pool.len() {
        pool[index] = hash_mix(entropy.get(index).copied().unwrap_or(0), &mut hash_const);
    }
    for source in 0..pool.len() {
        let source_value = pool[source];
        for (destination, value) in pool.iter_mut().enumerate() {
            if source != destination {
                let mixed = hash_mix(source_value, &mut hash_const);
                *value = mix(*value, mixed);
            }
        }
    }
    for &extra in entropy.iter().skip(pool.len()) {
        for value in &mut pool {
            let mixed = hash_mix(extra, &mut hash_const);
            *value = mix(*value, mixed);
        }
    }
    pool
}

fn hash_mix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> 16)
}

fn mix(left: u32, right: u32) -> u32 {
    let value = MIX_MULT_L
        .wrapping_mul(left)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(right));
    value ^ (value >> 16)
}

fn random_log_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let n = if x < 7.0 { (7.0 - x) as u64 } else { 0 };
    let mut x0 = x + n as f64;
    let x2 = (1.0 / x0) * (1.0 / x0);
    let mut correction = COEFFICIENTS[9];
    for coefficient in COEFFICIENTS[..9].iter().rev() {
        correction *= x2;
        correction += coefficient;
    }
    let mut result = correction / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 1..=n {
            result -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::NumpyRng;

    #[test]
    fn pcg64_raw_stream_matches_numpy_2_2_6() {
        let mut rng = NumpyRng::new(91);
        assert_eq!(
            (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>(),
            vec![
                5_221_960_716_401_725_447,
                16_791_166_746_794_900_714,
                15_113_692_621_735_265_246,
                5_396_149_944_436_547_962,
                15_400_337_095_808_520_742,
            ]
        );
    }

    #[test]
    fn seed_sequence_matches_numpy_zero_and_max_u64_seeds() {
        let mut zero = NumpyRng::new(0);
        assert_eq!(
            (0..5).map(|_| zero.next_u64()).collect::<Vec<_>>(),
            vec![
                11_749_869_230_777_074_271,
                4_976_686_463_289_251_617,
                755_828_109_848_996_024,
                304_881_062_738_325_533,
                15_002_187_965_291_974_971,
            ]
        );
        let mut maximum = NumpyRng::new(u64::MAX);
        assert_eq!(
            (0..5).map(|_| maximum.next_u64()).collect::<Vec<_>>(),
            vec![
                12_544_278_110_101_001_871,
                15_593_249_672_699_323_225,
                136_562_751_618_339_402,
                16_501_869_284_920_798_641,
                2_378_988_675_683_841_820,
            ]
        );
    }

    #[test]
    fn poisson_and_bounded_integer_streams_match_numpy_2_2_6() {
        let mut poisson = NumpyRng::new(91);
        assert_eq!(
            (0..8)
                .map(|_| poisson.poisson(3.0).unwrap())
                .collect::<Vec<_>>(),
            vec![5, 4, 4, 2, 2, 4, 3, 6]
        );

        let mut integers = NumpyRng::new(91);
        assert_eq!(
            (0..8)
                .map(|_| integers.integer(3).unwrap())
                .collect::<Vec<_>>(),
            vec![2, 0, 1, 2, 0, 2, 2, 0]
        );
    }

    #[test]
    fn poisson_ptrs_matches_numpy_2_2_6() {
        let expected = [
            (10.0, vec![8, 13, 8, 9, 7, 14, 10, 13, 16, 9, 8, 10]),
            (15.0, vec![19, 14, 11, 20, 15, 19, 22, 14, 13, 18, 16, 9]),
            (50.0, vec![45, 57, 45, 48, 43, 60, 51, 57, 63, 48, 46, 56]),
            (
                256.0,
                vec![245, 273, 246, 251, 240, 278, 258, 271, 286, 252, 247, 270],
            ),
        ];
        for (lambda, expected) in expected {
            let mut rng = NumpyRng::new(91);
            let observed = (0..expected.len())
                .map(|_| rng.poisson(lambda).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(observed, expected, "lambda={lambda}");
        }
    }

    #[test]
    fn singleton_integer_range_does_not_advance_numpy_stream() {
        let mut rng = NumpyRng::new(91);
        assert_eq!(rng.integer(1), Some(0));
        assert_eq!(rng.next_u64(), 5_221_960_716_401_725_447);
    }
}
