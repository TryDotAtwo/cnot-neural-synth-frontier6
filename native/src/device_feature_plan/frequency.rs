use crate::cuda::FrequencyArgs;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{FeatureSpec, FrequencyContext};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy)]
pub struct FrequencyPlanInputs<'a> {
    pub contexts: &'a [FrequencyContext],
    pub max_width: u32,
    pub max_frequency_bits: usize,
}
struct PreparedFrequency {
    roles: Tensor,
    internal: Tensor,
    statistics: Tensor,
}
pub(super) struct FrequencyPlan {
    dimension: u32,
    device_index: usize,
    unavailable: Option<DeviceFeatureStatus>,
    prepared: Option<PreparedFrequency>,
}
fn allocate(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, String> {
    let count = shape
        .iter()
        .try_fold(
            1i64,
            |count, &n| if n < 0 { None } else { count.checked_mul(n) },
        )
        .ok_or("I07 tensor extent overflow")?;
    count
        .checked_mul(match kind {
            Kind::Int64 => 8,
            Kind::Uint8 => 1,
            _ => 4,
        })
        .ok_or("I07 tensor byte extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}
impl FrequencyPlan {
    pub(super) fn new(
        dimension: u32,
        device_index: usize,
        inputs: FrequencyPlanInputs<'_>,
    ) -> Result<Self, String> {
        let device = Device::Cuda(device_index);
        let context = inputs.contexts.iter().find(|c| c.dimension() == dimension);
        let unavailable = if let Some(context) = context {
            let bits = context.sources().len().checked_mul(context.targets().len());
            if dimension > inputs.max_width
                || bits.is_none_or(|bits| bits > inputs.max_frequency_bits)
            {
                Some(DeviceFeatureStatus::ResourceLimit)
            } else {
                None
            }
        } else {
            Some(DeviceFeatureStatus::Unsupported)
        };
        let prepared = if unavailable.is_none() {
            let c = context.expect("matched frequency context");
            let mut roles = vec![0u8; dimension as usize];
            for &wire in c.internal() {
                roles[wire as usize] = 1;
            }
            for &wire in c.targets() {
                roles[wire as usize] = 2;
            }
            let roles = Tensor::from_slice(&roles).to_device(device);
            let internal =
                Tensor::from_slice(&c.internal().iter().map(|&i| i as i32).collect::<Vec<_>>())
                    .to_device(device);
            let central = Tensor::from_slice(
                &c.central_rows()
                    .iter()
                    .flatten()
                    .map(|&v| v as i64)
                    .collect::<Vec<_>>(),
            )
            .view([
                c.targets().len() as i64,
                c.sources().len().div_ceil(64) as i64,
            ])
            .to_device(device);
            let scratch = allocate(&central.size(), Kind::Int64, device)?;
            let statistics = allocate(&[2], Kind::Int64, device)?;
            // Even immutable numerical features are computed on CUDA.
            unsafe {
                crate::cuda::prepare_frequency(
                    central.data_ptr().cast(),
                    scratch.data_ptr().cast(),
                    statistics.data_ptr().cast(),
                    c.targets().len() as u32,
                    c.sources().len() as u32,
                    c.internal().len() as u32,
                    scratch.numel() as u64,
                    crate::cuda::current_stream(device_index).map_err(str::to_owned)?,
                )?;
            }
            Some(PreparedFrequency {
                roles,
                internal,
                statistics,
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
            return Err("I07 plan and states have different widths/devices".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("I07 requires an explicit state alphabet identity")?;
        let unavailable = if alphabet != format!("all_to_all_unit_cnot:{}", self.dimension) {
            Some(DeviceFeatureStatus::Unsupported)
        } else {
            self.unavailable
        };
        let b = states.batch_size();
        let mut status = allocate(&[b], Kind::Uint8, device)?;
        let mut values = allocate(&[b, 2], Kind::Float, device)?;
        if let Some(code) = unavailable {
            status = status.full_like(code as i64);
            values = values.zeros_like();
        } else {
            let p = self.prepared.as_ref().expect("prepared frequency context");
            let t = p.internal.size()[0];
            let scratch = allocate(&[b, t, (t + 63) / 64], Kind::Int64, device)?;
            let args = FrequencyArgs {
                states: states.tensor().data_ptr().cast(),
                roles: p.roles.data_ptr().cast(),
                internal_wires: p.internal.data_ptr().cast(),
                statistics: p.statistics.data_ptr().cast(),
                scratch: scratch.data_ptr().cast(),
                status: status.data_ptr().cast(),
                values: values.data_ptr().cast(),
                stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
                scratch_words: scratch.numel() as u64,
                batch: u32::try_from(b).map_err(|_| "I07 batch exceeds CUDA range")?,
                dimension: self.dimension,
                words_per_row: self.dimension.div_ceil(64),
                internal_count: t as u32,
            };
            unsafe {
                crate::cuda::launch_frequency(&args)?;
            }
        }
        let spec = specs
            .iter()
            .find(|s| (s.id.family(), s.id.ordinal()) == ('I', 7))
            .expect("selected I07");
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
