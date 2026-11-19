use super::{DeviceFeatureEntry, DeviceFeaturePacket, DeviceFeatureStatus};
use crate::model::tensor_packet::TensorPacket;
use tch::{Kind, Tensor};

impl DeviceFeaturePacket {
    /// One device integer per canonical group feature, across this entire batch.
    pub(crate) fn source_group_widths(&self) -> Tensor {
        let widths: Vec<_> = self
            .tensors
            .entries
            .iter()
            .flatten()
            .filter_map(|e| {
                e.token_valid.as_ref().map(|m| {
                    m.to_kind(Kind::Bool)
                        .logical_and(
                            &e.status
                                .eq(DeviceFeatureStatus::Computed as i64)
                                .unsqueeze(1),
                        )
                        .sum_dim_intlist([1i64].as_slice(), false, Kind::Int64)
                        .max()
                })
            })
            .collect();
        if widths.is_empty() {
            Tensor::empty([0], (Kind::Int64, self.matrix().device()))
        } else {
            Tensor::stack(&widths, 0)
        }
    }

    /// Stable-compaction into the exact logical source axes. The source uploads
    /// contiguous packet arrays. Reproduce their strides as well as their values.
    pub(crate) fn source_layout(&self, widths: &[i64]) -> Result<Self, String> {
        let mut widths = widths.iter().copied();
        let mut entries = Vec::with_capacity(self.tensors.entries.len());
        for slot in &self.tensors.entries {
            let Some(e) = slot else {
                entries.push(None);
                continue;
            };
            let new = if let Some(mask) = &e.token_valid {
                let width = widths.next().ok_or("missing source group extent")?;
                if width < 0 || width > mask.size()[1] {
                    return Err("invalid source group extent".into());
                }
                let mask = mask.to_kind(Kind::Bool).logical_and(
                    &e.status
                        .eq(DeviceFeatureStatus::Computed as i64)
                        .unsqueeze(1),
                );
                let order = mask
                    .to_kind(Kind::Int64)
                    .argsort_stable(true, 1, true)
                    .narrow(1, 0, width);
                let mask = mask.gather(1, &order, false);
                let compact = |t: &Tensor| {
                    let mut axes = t.size();
                    axes[1] = width;
                    let mut index = order.shallow_clone();
                    let mut valid = mask.shallow_clone();
                    for _ in 2..axes.len() {
                        index = index.unsqueeze(-1);
                        valid = valid.unsqueeze(-1);
                    }
                    let out = t.gather(1, &index.expand(axes.as_slice(), true), false);
                    out.masked_fill(&valid.logical_not(), 0.0).contiguous()
                };
                DeviceFeatureEntry::new(
                    e.spec.clone(),
                    compact(&e.values),
                    e.status.contiguous(),
                    Some(mask.shallow_clone()),
                    Some(compact(
                        e.incidence.as_ref().expect("validated group incidence"),
                    )),
                )
            } else {
                DeviceFeatureEntry::new(
                    e.spec.clone(),
                    e.values.contiguous(),
                    e.status.contiguous(),
                    None,
                    None,
                )
            };
            entries.push(Some(new));
        }
        if widths.next().is_some() {
            return Err("extra source group extent".into());
        }
        Ok(Self {
            tensors: TensorPacket {
                matrix: self.matrix().contiguous(),
                valid_nodes: self.valid_nodes().contiguous(),
                entries,
            },
            schema_id: self.schema_id.clone(),
            policy_id: self.policy_id.clone(),
        })
    }
}
