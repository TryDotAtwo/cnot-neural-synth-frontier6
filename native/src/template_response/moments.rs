//! Exact rational expectations, conditional rounding, and mixed differences.

use super::RationalProbability;

pub(super) fn expected_cost_fraction(
    costs: &[u64],
    probabilities: &[RationalProbability],
) -> Option<(u128, u128)> {
    if costs.len() != (1usize.checked_shl(probabilities.len() as u32)?) {
        return None;
    }
    let denominator = probabilities
        .iter()
        .try_fold(1u128, |product, probability| {
            product.checked_mul(u128::from(probability.denominator))
        })?;
    let mut numerator = 0u128;
    for (assignment, cost) in costs.iter().copied().enumerate() {
        let mut weight = 1u128;
        for (index, probability) in probabilities.iter().enumerate() {
            let factor = if assignment & (1usize << index) != 0 {
                probability.numerator
            } else {
                probability.denominator - probability.numerator
            };
            weight = weight.checked_mul(u128::from(factor))?;
        }
        numerator = numerator.checked_add(weight.checked_mul(u128::from(cost))?)?;
    }
    Some((numerator, denominator))
}

pub(super) fn conditional_choice_totals(
    costs: &[u64],
    probabilities: &[RationalProbability],
    fixed_index: usize,
) -> Option<(u128, u128)> {
    if probabilities.len() >= usize::BITS as usize
        || fixed_index >= probabilities.len()
        || costs.len() != (1usize.checked_shl(probabilities.len() as u32)?)
    {
        return None;
    }
    let mut zero_total = 0u128;
    let mut one_total = 0u128;
    for assignment in 0..costs.len() {
        if assignment & (1usize << fixed_index) != 0 {
            continue;
        }
        let mut weight = 1u128;
        for (index, probability) in probabilities.iter().enumerate() {
            if index == fixed_index {
                continue;
            }
            let factor = if assignment & (1usize << index) != 0 {
                probability.numerator
            } else {
                probability.denominator - probability.numerator
            };
            weight = weight.checked_mul(u128::from(factor))?;
        }
        zero_total = zero_total.checked_add(weight.checked_mul(u128::from(costs[assignment]))?)?;
        one_total = one_total.checked_add(
            weight.checked_mul(u128::from(costs[assignment | (1usize << fixed_index)]))?,
        )?;
    }
    Some((zero_total, one_total))
}

pub(super) fn conditional_round_cost(
    costs: &[u64],
    probabilities: &[RationalProbability],
) -> Option<u64> {
    let mut point = probabilities.to_vec();
    for index in 0..point.len() {
        let (zero_cost, one_cost) = conditional_choice_totals(costs, &point, index)?;
        point[index] = deterministic_probability(one_cost < zero_cost);
    }
    let assignment = point
        .iter()
        .enumerate()
        .fold(0usize, |mask, (index, probability)| {
            mask | (usize::from(probability.numerator == 1) << index)
        });
    costs.get(assignment).copied()
}

pub(super) fn mixed_difference_at_midpoint(
    costs: &[u64],
    first: usize,
    second: usize,
) -> Option<(i128, u128)> {
    let bit_count = costs.len().trailing_zeros() as usize;
    if costs.len() != (1usize.checked_shl(bit_count as u32)?)
        || first >= bit_count
        || second >= bit_count
        || first >= second
        || bit_count < 2
    {
        return None;
    }
    let assignment_count = 1usize.checked_shl((bit_count - 2) as u32)?;
    let mut difference_sum = 0i128;
    for reduced in 0..assignment_count {
        let mut base = 0usize;
        let mut reduced_index = 0usize;
        for index in 0..bit_count {
            if index == first || index == second {
                continue;
            }
            base |= ((reduced >> reduced_index) & 1) << index;
            reduced_index += 1;
        }
        let first_bit = 1usize << first;
        let second_bit = 1usize << second;
        let difference = i128::from(costs[base | first_bit | second_bit])
            - i128::from(costs[base | first_bit])
            - i128::from(costs[base | second_bit])
            + i128::from(costs[base]);
        difference_sum = difference_sum.checked_add(difference)?;
    }
    Some((difference_sum, assignment_count as u128))
}

pub(super) fn deterministic_probability(bit: bool) -> RationalProbability {
    RationalProbability {
        numerator: u64::from(bit),
        denominator: 1,
    }
}
