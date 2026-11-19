//! Stable padded scoring. Admission uses the number of live rows in each
//! source inference batch. No callback sees a real state from an over-budget
//! packet, and the decision never leaves the GPU.
use crate::model::DeviceFeatureStatus;
use tch::{Kind, Tensor};

pub(super) struct ScoringBatch<'a> {
    pub(super) width: i64,
    pub(super) capacity: i64,
    pub(super) identity: &'a Tensor,
}

impl ScoringBatch<'_> {
    pub(super) fn score(
        &self,
        rows: &Tensor,
        active: &Tensor,
        columns: Option<i64>,
        scorer: &mut impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
    ) -> Result<(Tensor, Tensor), String> {
        let mut parts = Vec::new();
        let mut coverage = Vec::new();
        let count = rows.size()[0];
        let mut start = 0;
        while start < count {
            let step = self.width.min(count - start);
            let batch = rows.narrow(0, start, step);
            let (values, votes) = if step <= self.capacity {
                checked(&batch, columns, scorer)?
            } else {
                self.limited(&batch, &active.narrow(0, start, step), columns, scorer)?
            };
            parts.push(values.to_kind(Kind::Double));
            coverage.push(votes);
            start += step;
        }
        Ok((Tensor::cat(&parts, 0), Tensor::cat(&coverage, 0)))
    }

    fn limited(
        &self,
        rows: &Tensor,
        active: &Tensor,
        columns: Option<i64>,
        scorer: &mut impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
    ) -> Result<(Tensor, Tensor), String> {
        let count = rows.size()[0];
        let device = rows.device();
        let mut values = Tensor::zeros(shape(count, columns), (Kind::Double, device));
        // The scalar packet rejects every requested input at its workspace
        // gate. A01 is the first input in the complete trained-model contract.
        let failure = 256 + DeviceFeatureStatus::ResourceLimit as i64;
        let rejected = Tensor::full([count], failure, (Kind::Int64, device));
        if self.capacity == 0 {
            return Ok((values, rejected));
        }
        let fits = active.sum(Kind::Int64).le(self.capacity);
        let first = active
            .to_kind(Kind::Int64)
            .argsort_stable(true, 0, true)
            .narrow(0, 0, self.capacity);
        let admitted = active
            .index_select(0, &first)
            .logical_and(&fits)
            .view([-1, 1, 1]);
        let packed = rows
            .index_select(0, &first)
            .where_self(&admitted, self.identity);
        let (scores, votes) = checked(&packed, columns, scorer)?;
        values = values.index_copy(0, &first, &scores.to_kind(Kind::Double));
        let coverage = rejected
            .index_copy(0, &first, &votes)
            .where_self(&fits, &rejected);
        Ok((values, coverage))
    }
}

fn shape(rows: i64, columns: Option<i64>) -> Vec<i64> {
    match columns {
        Some(n) => vec![rows, n],
        None => vec![rows],
    }
}

fn checked(
    rows: &Tensor,
    columns: Option<i64>,
    scorer: &mut impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
) -> Result<(Tensor, Tensor), String> {
    let (values, coverage) = scorer(rows)?;
    let count = rows.size()[0];
    if coverage.kind() != Kind::Int64
        || coverage.device() != rows.device()
        || coverage.size() != [count]
        || values.device() != rows.device()
        || values.size() != shape(count, columns)
        || !matches!(
            values.kind(),
            Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16
        )
    {
        return Err("search scorer returned a wrong shape, dtype or device".into());
    }
    Ok((values, coverage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn padded_search_charges_only_live_rows_and_substitutes_identity_on_rejection() {
        crate::torch_device::require_cuda_device(0).unwrap();
        let device = Device::Cuda(0);
        let identity = Tensor::from_slice(&[1i64, 2])
            .view([1, 2, 1])
            .to_device(device);
        let rows = Tensor::from_slice(&[2i64, 3, 3, 2, 1, 3, 3, 1, 2, 1, 1, 3, 3, 2, 2, 3, 1, 2])
            .view([9, 2, 1])
            .to_device(device);
        let live = Tensor::from_slice(&[0i64, 1, 0, 1, 1, 1, 0, 1, 1])
            .to_device(device)
            .ne(0);
        let to_host = |t: &Tensor| {
            Vec::<i64>::try_from(t.to_kind(Kind::Int64).to_device(Device::Cpu).view([-1])).unwrap()
        };
        for columns in [None, Some(2)] {
            let mut observed = Vec::new();
            let mut scorer = |t: &Tensor| {
                observed.push(to_host(t));
                let sums = t.sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Double);
                let values = match columns {
                    None => sums,
                    Some(_) => Tensor::stack(&[&sums, &(-&sums)], 1),
                };
                Ok((values, Tensor::zeros([t.size()[0]], (Kind::Int64, device))))
            };
            let (values, coverage) = ScoringBatch {
                width: 4,
                capacity: 2,
                identity: &identity,
            }
            .score(&rows, &live, columns, &mut scorer)
            .unwrap();
            assert_eq!(
                observed,
                vec![vec![3, 2, 3, 1], vec![1, 2, 1, 2], vec![1, 2]]
            );
            assert_eq!(to_host(&coverage), [260, 0, 260, 0, 260, 260, 260, 260, 0]);
            let positive = if columns.is_some() {
                values.select(1, 0)
            } else {
                values
            };
            assert_eq!(to_host(&positive)[1], 5);
            assert_eq!(to_host(&positive)[3], 4);
            assert_eq!(to_host(&positive)[8], 3);
        }
        let (values, coverage) = ScoringBatch {
            width: 4,
            capacity: 0,
            identity: &identity,
        }
        .score(&rows, &live, None, &mut |_| {
            panic!("a zero-byte packet cannot run a provider")
        })
        .unwrap();
        assert_eq!(to_host(&values), [0; 9]);
        assert_eq!(to_host(&coverage), [260; 9]);
    }
}
