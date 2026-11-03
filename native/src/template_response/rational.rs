//! Exact rational probability normalization and correctly rounded `f32` conversion.

use crate::features::FeatureBuildError;

/// A normalized rational Bernoulli probability in the closed unit interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RationalProbability {
    pub(super) numerator: u64,
    pub(super) denominator: u64,
}

impl RationalProbability {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, FeatureBuildError> {
        if denominator == 0 || numerator > denominator {
            return Err(FeatureBuildError(
                "template response probability must be a rational in [0,1]".into(),
            ));
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn numerator(self) -> u64 {
        self.numerator
    }

    pub fn denominator(self) -> u64 {
        self.denominator
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.max(1)
}

pub(super) fn rational_to_f32(numerator: u128, denominator: u128) -> Option<f32> {
    if denominator == 0 {
        return None;
    }
    if numerator == 0 {
        return Some(0.0);
    }

    let mut exponent = bit_length(numerator) - bit_length(denominator);
    if ratio_below_power_of_two(numerator, denominator, exponent)? {
        exponent -= 1;
    }
    if exponent > 127 {
        return None;
    }

    let subnormal = exponent < -126;
    let fractional_bits = if subnormal { 149 } else { 23 - exponent };
    if fractional_bits < 0 {
        return None;
    }
    let integer_part = numerator / denominator;
    let remainder = numerator % denominator;
    let whole = if integer_part == 0 {
        0
    } else {
        integer_part.checked_mul(1u128.checked_shl(fractional_bits as u32)?)?
    };
    let (fraction, remainder) =
        scaled_fraction_bits(remainder, denominator, fractional_bits as usize)?;
    let mut significand = whole.checked_add(fraction)?;
    let complement = denominator.checked_sub(remainder)?;
    if remainder > complement || (remainder == complement && significand & 1 == 1) {
        significand = significand.checked_add(1)?;
    }

    if subnormal {
        if significand > (1u128 << 23) {
            return None;
        }
        return Some(f32::from_bits(significand as u32));
    }

    if significand >= (1u128 << 24) {
        significand >>= 1;
        exponent += 1;
    }
    if exponent > 127 || significand < (1u128 << 23) {
        return None;
    }
    let exponent_bits = u32::try_from(exponent + 127).ok()?;
    let fraction_bits = u32::try_from(significand - (1u128 << 23)).ok()?;
    Some(f32::from_bits((exponent_bits << 23) | fraction_bits))
}

pub(super) fn signed_rational_to_f32(numerator: i128, denominator: u128) -> Option<f32> {
    let magnitude = numerator.unsigned_abs();
    let value = rational_to_f32(magnitude, denominator)?;
    Some(if numerator < 0 { -value } else { value })
}

fn scaled_fraction_bits(
    mut remainder: u128,
    denominator: u128,
    count: usize,
) -> Option<(u128, u128)> {
    let mut value = 0u128;
    for _ in 0..count {
        value = value.checked_mul(2)?;
        let complement = denominator.checked_sub(remainder)?;
        if remainder >= complement {
            remainder = remainder.checked_sub(complement)?;
            value = value.checked_add(1)?;
        } else {
            remainder = remainder.checked_mul(2)?;
        }
    }
    Some((value, remainder))
}

fn bit_length(value: u128) -> i32 {
    (u128::BITS - value.leading_zeros()) as i32
}

fn ratio_below_power_of_two(numerator: u128, denominator: u128, exponent: i32) -> Option<bool> {
    if exponent >= 0 {
        let scale = 1u128.checked_shl(exponent as u32)?;
        Some(match denominator.checked_mul(scale) {
            Some(threshold) => numerator < threshold,
            None => true,
        })
    } else {
        let scale = 1u128.checked_shl(exponent.unsigned_abs())?;
        Some(match numerator.checked_mul(scale) {
            Some(scaled) => scaled < denominator,
            None => false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::rational_to_f32;

    #[test]
    fn rational_to_f32_uses_round_to_nearest_ties_to_even() {
        let halfway_above_one = ((1u128 << 24) + 1, 1u128 << 24);
        let halfway_above_odd_significand = ((1u128 << 24) + 3, 1u128 << 24);

        assert_eq!(
            rational_to_f32(halfway_above_one.0, halfway_above_one.1)
                .unwrap()
                .to_bits(),
            1.0f32.to_bits()
        );
        assert_eq!(
            rational_to_f32(
                halfway_above_odd_significand.0,
                halfway_above_odd_significand.1
            )
            .unwrap()
            .to_bits(),
            0x3f80_0002
        );
    }
}
