use crate::cuda::SharedRelationArgs;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{FeatureSpec, SharedRelationContext};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy)]
pub struct SharedRelationPlanInputs<'a> {
    pub contexts: &'a [SharedRelationContext],
    pub max_vectors: usize,
    pub max_coordinates: usize,
    pub max_ports: usize,
}
struct PreparedRelation {
    rows: Tensor,
    coordinates: Tensor,
    covectors: Tensor,
}
pub(super) struct SharedRelationPlan {
    dimension: u32,
    device_index: usize,
    max_vectors: i64,
    capacity: i64,
    unavailable: Option<DeviceFeatureStatus>,
    prepared: Option<PreparedRelation>,
}
fn allocate(shape: &[i64], kind: Kind, device: Device, zero: bool) -> Result<Tensor, String> {
    let count = shape
        .iter()
        .try_fold(
            1i64,
            |count, &n| if n < 0 { None } else { count.checked_mul(n) },
        )
        .ok_or("I06 tensor extent overflow")?;
    count
        .checked_mul(match kind {
            Kind::Int64 => 8,
            Kind::Uint8 => 1,
            _ => 4,
        })
        .ok_or("I06 tensor byte extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl SharedRelationPlan {
    pub(super) fn new(
        dimension: u32,
        device_index: usize,
        input: SharedRelationPlanInputs<'_>,
        max_bytes: Option<usize>,
    ) -> Result<Self, String> {
        let context = input.contexts.iter().find(|c| c.dimension() == dimension);
        let unavailable = if let Some(c) = context {
            let estimate = crate::shared_relation::working_bytes_upper_bound(
                dimension as usize,
                c.rows().len(),
                c.coordinates().len(),
                c.covectors().len(),
                input.max_vectors,
            );
            if c.rows().len() > input.max_vectors
                || c.coordinates().len() > input.max_coordinates
                || c.covectors().len() > input.max_ports
                || estimate.is_none_or(|value| {
                    value > max_bytes.unwrap_or(crate::shared_relation::DEFAULT_MAX_BYTES)
                })
            {
                Some(DeviceFeatureStatus::ResourceLimit)
            } else {
                None
            }
        } else {
            Some(DeviceFeatureStatus::Unsupported)
        };
        let mut capacity = 0;
        let prepared = if unavailable.is_none() {
            let c = context.expect("matched relation context");
            let device = Device::Cuda(device_index);
            let rows = Tensor::from_slice(&c.rows().iter().map(|&v| v as i32).collect::<Vec<_>>())
                .to_device(device);
            let coordinates = Tensor::from_slice(
                &c.coordinates()
                    .iter()
                    .map(|&v| v as i32)
                    .collect::<Vec<_>>(),
            )
            .to_device(device);
            let w = dimension.div_ceil(64) as usize;
            let mut physical = vec![
                0i64;
                c.covectors()
                    .len()
                    .checked_mul(w)
                    .ok_or("I06 physical covector extent overflow")?
            ];
            // Pack only immutable caller-supplied covector labels. Their
            // products with matrix rows are always computed on the device.
            for (label, covector) in c.covectors().iter().enumerate() {
                for (index, &wire) in c.coordinates().iter().enumerate() {
                    if covector[index / 64] & (1u64 << (index % 64)) != 0 {
                        physical[label * w + wire as usize / 64] |= (1u64 << (wire % 64)) as i64;
                    }
                }
            }
            let covectors = Tensor::from_slice(&physical)
                .view([c.covectors().len() as i64, w as i64])
                .to_device(device);
            capacity = (c.rows().len() + c.covectors().len()) as i64;
            Some(PreparedRelation {
                rows,
                coordinates,
                covectors,
            })
        } else {
            None
        };
        Ok(Self {
            dimension,
            device_index,
            max_vectors: input.max_vectors as i64,
            capacity,
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
            return Err("I06 plan and states have different widths/devices".into());
        }
        let b = states.batch_size();
        let g = self.capacity;
        let values = allocate(
            &[b, g, self.max_vectors + 2],
            Kind::Float,
            device,
            self.unavailable.is_some(),
        )?;
        let valid = allocate(&[b, g], Kind::Uint8, device, self.unavailable.is_some())?;
        let incidence = allocate(
            &[b, g, i64::from(self.dimension), 2],
            Kind::Float,
            device,
            self.unavailable.is_some(),
        )?;
        let mut status = allocate(&[b], Kind::Uint8, device, false)?;
        if let Some(code) = self.unavailable {
            status = status.full_like(code as i64);
        } else {
            let p = self.prepared.as_ref().expect("prepared shared relation");
            let v = p.rows.size()[0];
            let c = p.coordinates.size()[0];
            let scratch = allocate(
                &[b, (c + v) * ((v + 63) / 64) + v],
                Kind::Int64,
                device,
                false,
            )?;
            let args = SharedRelationArgs {
                states: states.tensor().data_ptr().cast(),
                row_wires: p.rows.data_ptr().cast(),
                coordinate_wires: p.coordinates.data_ptr().cast(),
                covectors: p.covectors.data_ptr().cast(),
                scratch: scratch.data_ptr().cast(),
                values: values.data_ptr().cast(),
                valid: valid.data_ptr().cast(),
                incidence: incidence.data_ptr().cast(),
                status: status.data_ptr().cast(),
                stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
                scratch_words: scratch.numel() as u64,
                value_elements: values.numel() as u64,
                group_elements: valid.numel() as u64,
                incidence_elements: incidence.numel() as u64,
                batch: u32::try_from(b).map_err(|_| "I06 batch exceeds CUDA range")?,
                dimension: self.dimension,
                words_per_row: self.dimension.div_ceil(64),
                vectors: v as u32,
                coordinates: c as u32,
                ports: p.covectors.size()[0] as u32,
                max_vectors: self.max_vectors as u32,
            };
            unsafe {
                crate::cuda::launch_shared_relation(&args)?;
            }
        }
        let spec = specs
            .iter()
            .find(|s| (s.id.family(), s.id.ordinal()) == ('I', 6))
            .expect("selected I06");
        entries.push(DeviceFeatureEntry::new(
            spec.clone(),
            values,
            status,
            Some(valid),
            Some(incidence),
        ));
        Ok(())
    }
}
