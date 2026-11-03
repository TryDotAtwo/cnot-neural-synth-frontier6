use super::NumpyRng;

impl NumpyRng {
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
