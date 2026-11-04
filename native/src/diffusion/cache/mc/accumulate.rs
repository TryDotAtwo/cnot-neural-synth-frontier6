//! Joint sample moments and uncertainty summaries.

use super::JointMoments;

pub(super) struct MomentAccumulator {
    width: usize,
    sum_weight: Vec<f64>,
    sum_weight_pair: Vec<f64>,
    sum_weight_intersection: Vec<f64>,
    sum_weight_xor: Vec<f64>,
    sum_intersection: Vec<f64>,
    sum_intersection_squared: Vec<f64>,
    sum_xor: Vec<f64>,
    sum_xor_squared: Vec<f64>,
    sum_character: Vec<f64>,
    sum_character_pair: Vec<f64>,
    sum_character_pair_squared: Vec<f64>,
}

impl MomentAccumulator {
    pub(super) fn new(width: usize) -> Option<Self> {
        let pair_count = width.checked_mul(width)?;
        Some(Self {
            width,
            sum_weight: vec![0.0; width],
            sum_weight_pair: vec![0.0; pair_count],
            sum_weight_intersection: vec![0.0; pair_count],
            sum_weight_xor: vec![0.0; pair_count],
            sum_intersection: vec![0.0; pair_count],
            sum_intersection_squared: vec![0.0; pair_count],
            sum_xor: vec![0.0; pair_count],
            sum_xor_squared: vec![0.0; pair_count],
            sum_character: vec![0.0; width],
            sum_character_pair: vec![0.0; pair_count],
            sum_character_pair_squared: vec![0.0; pair_count],
        })
    }

    pub(super) fn observe_columns(&mut self, columns: &[Vec<u64>]) {
        let n = self.width;
        let weights = columns
            .iter()
            .map(|column| column.iter().map(|word| word.count_ones() as f64).sum())
            .collect::<Vec<f64>>();
        let characters = weights
            .iter()
            .map(|weight| {
                if (*weight as usize) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<f64>>();
        for first in 0..n {
            self.sum_weight[first] += weights[first];
            self.sum_character[first] += characters[first];
            for second in 0..n {
                let index = first * n + second;
                self.sum_weight_pair[index] += weights[first] * weights[second];
                let mut intersection = 0u32;
                let mut xor_weight = 0u32;
                for word in 0..columns[first].len() {
                    let left = columns[first][word];
                    let right = columns[second][word];
                    intersection += (left & right).count_ones();
                    xor_weight += (left ^ right).count_ones();
                }
                let intersection = f64::from(intersection);
                let xor_weight = f64::from(xor_weight);
                let character_pair = characters[first] * characters[second];
                self.sum_intersection[index] += intersection;
                self.sum_intersection_squared[index] += intersection * intersection;
                self.sum_xor[index] += xor_weight;
                self.sum_xor_squared[index] += xor_weight * xor_weight;
                self.sum_weight_intersection[index] += weights[first] * intersection;
                self.sum_weight_xor[index] += weights[first] * xor_weight;
                self.sum_character_pair[index] += character_pair;
                self.sum_character_pair_squared[index] += character_pair * character_pair;
            }
        }
    }

    pub(super) fn finish(self, samples: usize, total_gates: u64) -> JointMoments {
        let n = self.width;
        let pair_count = n * n;
        let denominator = (samples - 1) as f64;
        let mean_weight = self
            .sum_weight
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_intersection = self
            .sum_intersection
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_xor = self
            .sum_xor
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_character = self
            .sum_character
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();
        let mean_character_pair = self
            .sum_character_pair
            .iter()
            .map(|sum| sum / samples as f64)
            .collect::<Vec<_>>();

        let mut covariance_weight = vec![0.0; pair_count];
        let mut covariance_weight_intersection = vec![0.0; pair_count];
        let mut covariance_weight_xor = vec![0.0; pair_count];
        let mut covariance_character_pair = vec![0.0; pair_count];
        let mut standard_error_intersection = vec![0.0; pair_count];
        let mut standard_error_xor = vec![0.0; pair_count];
        let mut standard_error_character_pair = vec![0.0; pair_count];
        for first in 0..n {
            for second in 0..n {
                let index = first * n + second;
                covariance_weight[index] = (self.sum_weight_pair[index]
                    - self.sum_weight[first] * mean_weight[second])
                    / denominator;
                covariance_weight_intersection[index] = (self.sum_weight_intersection[index]
                    - self.sum_weight[first] * mean_intersection[index])
                    / denominator;
                covariance_weight_xor[index] = (self.sum_weight_xor[index]
                    - self.sum_weight[first] * mean_xor[index])
                    / denominator;
                covariance_character_pair[index] =
                    mean_character_pair[index] - mean_character[first] * mean_character[second];
                standard_error_intersection[index] = standard_error(
                    self.sum_intersection[index],
                    self.sum_intersection_squared[index],
                    samples,
                );
                standard_error_xor[index] =
                    standard_error(self.sum_xor[index], self.sum_xor_squared[index], samples);
                standard_error_character_pair[index] = standard_error(
                    self.sum_character_pair[index],
                    self.sum_character_pair_squared[index],
                    samples,
                );
            }
        }

        JointMoments {
            samples,
            total_gates,
            mean_weight,
            covariance_weight,
            covariance_weight_intersection,
            covariance_weight_xor,
            mean_intersection,
            mean_xor,
            mean_character_pair,
            covariance_character_pair,
            standard_error_intersection,
            standard_error_xor,
            standard_error_character_pair,
        }
    }
}

fn standard_error(sum: f64, sum_squared: f64, samples: usize) -> f64 {
    let samples_f64 = samples as f64;
    let variance = ((sum_squared - sum * sum / samples_f64) / (samples - 1) as f64).max(0.0);
    variance.sqrt() / samples_f64.sqrt()
}
