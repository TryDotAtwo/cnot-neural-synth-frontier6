//! Group extents for a sealed parent and its immutable successor chunks.
//! These extents select the same FP32 GEMM/attention shapes as the reference.
//! Only layout metadata crosses to the host once during admission.
use super::{DevicePrimitiveBellman, RootBranch};
use crate::{device_feature_plan::DeviceFeaturePlan, device_state::DeviceStateBatch};
use tch::{Device, Tensor};

pub(crate) struct RecordLayout {
    pub(super) parent: Vec<i64>,
    pub(super) signatures: Vec<Vec<i64>>,
    pub(super) chunk_signatures: Vec<usize>,
    pub(super) device_signatures: Tensor,
}

impl DevicePrimitiveBellman {
    pub(crate) fn prepare_layout(
        &self,
        root: &DeviceStateBatch,
        plan: &DeviceFeaturePlan,
        branch: RootBranch,
        chunk: usize,
    ) -> Result<RecordLayout, String> {
        let (count, width) = self.schedule(branch, chunk)?;
        let chunks = count.div_ceil(width);
        if !plan.has_groups() {
            return Ok(RecordLayout {
                parent: vec![],
                signatures: vec![vec![]],
                chunk_signatures: vec![0; chunks],
                device_signatures: Tensor::zeros(
                    [chunks as i64],
                    (tch::Kind::Int64, root.tensor().device()),
                ),
            });
        }
        let mut extents = Vec::with_capacity(1 + chunks);
        extents.push(plan.build(root)?.source_group_widths());
        for start in (0..count).step_by(width) {
            let children = self.children(root, start as i64, (count - start).min(width) as i64)?;
            extents.push(plan.build(&children)?.source_group_widths());
        }
        let host = Tensor::stack(&extents, 0).to_device(Device::Cpu);
        let mut layouts = Vec::<Vec<i64>>::try_from(&host)
            .map_err(|e| e.to_string())?
            .into_iter();
        let parent = layouts.next().ok_or("missing parent layout")?;
        let mut signatures = Vec::new();
        let mut chunk_signatures = Vec::new();
        for layout in layouts {
            let index = signatures
                .iter()
                .position(|s| *s == layout)
                .unwrap_or_else(|| {
                    signatures.push(layout);
                    signatures.len() - 1
                });
            chunk_signatures.push(index);
        }
        let ids: Vec<_> = chunk_signatures.iter().map(|&s| s as i64).collect();
        Ok(RecordLayout {
            parent,
            signatures,
            chunk_signatures,
            device_signatures: Tensor::from_slice(&ids).to_device(root.tensor().device()),
        })
    }

    pub(super) fn schedule(
        &self,
        branch: RootBranch,
        chunk: usize,
    ) -> Result<(usize, usize), String> {
        if chunk == 0 {
            return Err("positive neighbor chunk size required".into());
        }
        let actions = self.alphabet.gates().len();
        match branch {
            RootBranch::Nonterminal if actions == 0 => {
                Err("nonterminal state has an empty move alphabet".into())
            }
            RootBranch::Nonterminal => Ok((actions, chunk)),
            RootBranch::Terminal { qhat: true } => Ok((actions, actions.max(1))),
            RootBranch::Terminal { qhat: false } => Ok((0, 1)),
        }
    }
}
