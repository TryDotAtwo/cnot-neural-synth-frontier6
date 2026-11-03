const PCG_MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

#[derive(Clone, Debug)]
pub(crate) struct NumpyRng {
    state: u128,
    increment: u128,
    has_uint32: bool,
    cached_uint32: u32,
}

impl NumpyRng {
    pub(crate) fn new(seed: u64) -> Self {
        let (state, increment) = super::seed_sequence::state_and_increment(seed);

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

    pub(super) fn next_f64(&mut self) -> f64 {
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

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
    }
}
