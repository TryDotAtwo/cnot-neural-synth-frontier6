//! Selection of an explicitly approximate primitive-action subset.

use tch::{Kind, Tensor};

/// Select the lowest predicted action costs plus a deterministic rotating
/// exploration set. Output is in alphabet order, preserving first-tie rules.
/// A caller must separately vote on finite predictions and action masks.
pub(super) fn sampled_indices(scores: &Tensor, count: usize, seed: u64) -> Result<Tensor, String> {
    if scores.dim() != 1 || count < 2 {
        return Err("candidate sampling requires a 1-D score vector and at least two actions".into());
    }
    let size = scores.size()[0];
    if size <= 0 || size.checked_mul(size).is_none() {
        return Err("candidate alphabet has an invalid size".into());
    }
    let axis = Tensor::arange(size, (Kind::Int64, scores.device()));
    if count >= size as usize {
        return Ok(axis);
    }
    let exploitation = count.div_ceil(2);
    let exploration = count - exploitation;
    // Stable sorting resolves equal predicted costs in alphabet order.
    let preferred = scores
        .detach()
        .argsort_stable(true, 0, false)
        .narrow(0, 0, exploitation as i64);
    let mut stride = size / 2 + 1;
    while gcd(stride, size) != 1 {
        stride += 1;
    }
    let offset = (seed % size as u64) as i64;
    // Multiplication by a coprime stride permutes the action IDs. Excluding
    // preferred IDs before top-k makes exploration disjoint without a host sync.
    let priority = (&axis * stride + offset)
        .remainder(size)
        .scatter_value(0, &preferred, -1);
    let exploratory = priority.topk(exploration as i64, 0, true, true).1;
    Ok(Tensor::cat(&[preferred, exploratory], 0).sort(0, false).0)
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::sampled_indices;
    use std::collections::BTreeSet;
    use tch::{Device, Tensor};

    fn selected(scores: &Tensor, count: usize, seed: u64) -> Vec<i64> {
        Vec::<i64>::try_from(sampled_indices(scores, count, seed).unwrap().to_device(Device::Cpu))
            .unwrap()
    }

    #[test]
    fn shortlist_is_sorted_unique_reproducible_and_keeps_low_qhat_actions() {
        crate::torch_device::require_cuda_device(0).unwrap();
        let scores = Tensor::from_slice(&[8f32, 7., 6., 1., 5., 2., 9., 10., 11., 12., 0., 4.]);
        for device in [Device::Cpu, Device::Cuda(0)] {
            let scores = scores.to_device(device);
            let first = selected(&scores, 6, 42);
            assert_eq!(first, selected(&scores, 6, 42));
            assert_eq!(first.len(), 6);
            assert_eq!(first.iter().copied().collect::<BTreeSet<_>>().len(), 6);
            assert!(first.windows(2).all(|p| p[0] < p[1]));
            for index in [3, 5, 10] {
                assert!(first.contains(&index), "missing qhat action {index}");
            }
            assert_ne!(first, selected(&scores, 6, 43));
            assert_eq!(selected(&scores, 20, 42), (0..12).collect::<Vec<_>>());
        }
    }

    #[test]
    fn tied_qhat_actions_use_alphabet_order_and_invalid_caps_fail() {
        crate::torch_device::require_cuda_device(0).unwrap();
        let scores = Tensor::zeros([9], (tch::Kind::Float, Device::Cuda(0)));
        let indices = selected(&scores, 4, 12);
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(sampled_indices(&scores, 0, 0).is_err());
        assert!(sampled_indices(&scores, 1, 0).is_err());
        assert!(sampled_indices(&scores.view([3, 3]), 4, 0).is_err());
    }
}
