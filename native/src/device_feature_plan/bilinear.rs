use crate::cuda::{BilinearArgs, BilinearPrepareArgs};
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{BilinearContext, FeatureSpec};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy)]
pub struct BilinearPlanInputs<'a> {
    pub contexts: &'a [BilinearContext],
    pub max_dimension: usize,
    pub max_matrix_bits: usize,
}
struct PreparedBilinear {
    rows: Tensor,
    columns: Tensor,
    matrix_template: Tensor,
    statistics: Tensor,
    m: u32,
    r: u32,
    p: u32,
    q: u32,
}
pub(super) struct BilinearPlan {
    dimension: u32,
    device_index: usize,
    unavailable: Option<DeviceFeatureStatus>,
    prepared: Option<PreparedBilinear>,
}
fn allocate(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, String> {
    shape
        .iter()
        .try_fold(
            1i64,
            |count, &n| if n < 0 { None } else { count.checked_mul(n) },
        )
        .and_then(|count| {
            count.checked_mul(match kind {
                Kind::Int64 => 8,
                Kind::Uint8 => 1,
                _ => 4,
            })
        })
        .ok_or("I05 tensor extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}
fn upload(matrix: &[Vec<u64>], device: Device) -> Tensor {
    Tensor::from_slice(
        &matrix
            .iter()
            .flatten()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}
impl BilinearPlan {
    pub(super) fn new(
        dimension: u32,
        device_index: usize,
        inputs: BilinearPlanInputs<'_>,
        max_bytes: Option<usize>,
    ) -> Result<Self, String> {
        let context = inputs.contexts.iter().find(|c| c.dimension() == dimension);
        let unavailable = if let Some(c) = context {
            let (m, r, p, t, q) = c.dimensions();
            if [m, r, p, t, q].into_iter().max().unwrap_or(0) > inputs.max_dimension
                || c.matrix_bit_budget()
                    .is_none_or(|bits| bits > inputs.max_matrix_bits)
                || max_bytes.is_some_and(|cap| {
                    c.working_bytes_upper_bound()
                        .is_none_or(|bytes| bytes > cap)
                })
            {
                Some(DeviceFeatureStatus::ResourceLimit)
            } else {
                None
            }
        } else {
            Some(DeviceFeatureStatus::Unsupported)
        };
        let prepared = if unavailable.is_none() {
            let c = context.expect("matched bilinear context");
            let (m, r, p, t, q) = c.dimensions();
            let convert =
                |n| u32::try_from(n).map_err(|_| "I05 dimension exceeds CUDA index range");
            let (m, r, p, t, q) = (
                convert(m)?,
                convert(r)?,
                convert(p)?,
                convert(t)?,
                convert(q)?,
            );
            let rows_count = p.checked_add(m).ok_or("I05 combined row count overflow")?;
            let cols_count = q
                .checked_add(r)
                .ok_or("I05 combined column count overflow")?;
            let device = Device::Cuda(device_index);
            let rows = Tensor::from_slice(&c.rows().iter().map(|&v| v as i32).collect::<Vec<_>>())
                .to_device(device);
            let columns =
                Tensor::from_slice(&c.columns().iter().map(|&v| v as i32).collect::<Vec<_>>())
                    .to_device(device);
            let u = upload(c.u_rows(), device);
            let v = upload(c.v_rows(), device);
            let a0 = upload(c.a0_rows(), device);
            let d0 = upload(c.d0_rows(), device);
            let matrix_template = allocate(
                &[i64::from(rows_count), i64::from(cols_count.div_ceil(64))],
                Kind::Int64,
                device,
            )?;
            let scratch = allocate(&matrix_template.size(), Kind::Int64, device)?;
            let statistics = allocate(&[4], Kind::Int64, device)?;
            let args = BilinearPrepareArgs {
                u: u.data_ptr().cast(),
                v: v.data_ptr().cast(),
                a0: a0.data_ptr().cast(),
                d0: d0.data_ptr().cast(),
                matrix_template: matrix_template.data_ptr().cast(),
                scratch: scratch.data_ptr().cast(),
                statistics: statistics.data_ptr().cast(),
                stream: crate::cuda::current_stream(device_index).map_err(str::to_owned)?,
                template_words: matrix_template.numel() as u64,
                scratch_words: scratch.numel() as u64,
                m,
                r,
                p,
                t,
                q,
            };
            unsafe {
                crate::cuda::prepare_bilinear(&args)?;
            }
            Some(PreparedBilinear {
                rows,
                columns,
                matrix_template,
                statistics,
                m,
                r,
                p,
                q,
            })
        } else {
            None
        };
        Ok(Self {
            dimension,
            device_index,
            unavailable,
            prepared,
        })
    }
    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        entries: &mut Vec<DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("I05 plan and states have different widths/devices".into());
        }
        let b = states.batch_size();
        let mut values = allocate(&[b, 2], Kind::Float, device)?;
        let mut status = allocate(&[b], Kind::Uint8, device)?;
        if let Some(code) = self.unavailable {
            values = values.zeros_like();
            status = status.full_like(code as i64);
        } else {
            let p = self.prepared.as_ref().expect("prepared bilinear context");
            let scratch = allocate(
                &[b, p.matrix_template.size()[0], p.matrix_template.size()[1]],
                Kind::Int64,
                device,
            )?;
            let args = BilinearArgs {
                states: states.tensor().data_ptr().cast(),
                row_wires: p.rows.data_ptr().cast(),
                column_wires: p.columns.data_ptr().cast(),
                matrix_template: p.matrix_template.data_ptr().cast(),
                statistics: p.statistics.data_ptr().cast(),
                scratch: scratch.data_ptr().cast(),
                values: values.data_ptr().cast(),
                status: status.data_ptr().cast(),
                stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
                template_words: p.matrix_template.numel() as u64,
                scratch_words: scratch.numel() as u64,
                batch: u32::try_from(b).map_err(|_| "I05 batch exceeds CUDA range")?,
                dimension: self.dimension,
                words_per_row: self.dimension.div_ceil(64),
                m: p.m,
                r: p.r,
                p: p.p,
                q: p.q,
            };
            unsafe {
                crate::cuda::launch_bilinear(&args)?;
            }
        }
        let spec = specs
            .iter()
            .find(|s| (s.id.family(), s.id.ordinal()) == ('I', 5))
            .expect("selected I05");
        entries.push(DeviceFeatureEntry::new(
            spec.clone(),
            values,
            status,
            None,
            None,
        ));
        Ok(())
    }
}
