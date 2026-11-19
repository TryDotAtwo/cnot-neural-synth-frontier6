use crate::cuda::column_diffusion as ffi;
use crate::device_state::{DeviceFamilyA, DeviceStateBatch};
use crate::model::DeviceFeatureEntry;
use crate::{DiffusionConfig, FeatureSpec};
use tch::{Device, Kind, Tensor};

pub(super) struct ColumnDiffusion {
    config: ffi::Config,
    device_index: usize,
    stream: usize,
    descriptors: Tensor,
    status: Tensor,
}

impl ColumnDiffusion {
    pub(super) fn new(n: u32, index: usize, config: DiffusionConfig) -> Result<Self, String> {
        let config = ffi::Config {
            dimension: n,
            taus: config.taus,
            floor: config.floor,
            max_n: config.max_n,
        };
        let layout = ffi::layout(&config)?;
        let classes = i64::from(layout.classes);
        let device = Device::Cuda(index);
        let empty = |len, kind| Tensor::f_empty([len], (kind, device)).map_err(|e| e.to_string());
        let work = empty(18 * classes, Kind::Double)?;
        let predecessors = empty(4 * classes, Kind::Int)?;
        let weights = empty(4 * i64::from(layout.weight_stride), Kind::Double)?;
        let phases = empty(if classes == 0 { 0 } else { 5 }, Kind::Uint8)?;
        let descriptors = empty(47 * classes, Kind::Float)?;
        let status = empty(1, Kind::Uint8)?;
        let stream = crate::cuda::current_stream(index)?;
        let args = ffi::Prepare {
            config,
            work: work.data_ptr().cast(),
            predecessors: predecessors.data_ptr().cast(),
            weights: weights.data_ptr().cast(),
            phase_status: phases.data_ptr().cast(),
            descriptors: descriptors.data_ptr().cast(),
            status: status.data_ptr().cast(),
            stream,
            work_len: work.numel() as u64,
            predecessor_len: predecessors.numel() as u64,
            weight_len: weights.numel() as u64,
            phase_len: phases.numel() as u64,
            descriptor_len: descriptors.numel() as u64,
            status_len: status.numel() as u64,
        };
        // Setup scratch was allocated on this Torch stream. The allocator
        // reuses it in stream order after all three construction kernels.
        unsafe { ffi::prepare(&args)? };
        Ok(Self {
            config,
            device_index: index,
            stream: stream as usize,
            descriptors,
            status,
        })
    }

    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        family_a: &DeviceFamilyA,
        specs: &[FeatureSpec],
        entries: &mut Vec<DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let stream = crate::cuda::current_stream(self.device_index)?;
        if stream as usize != self.stream {
            return Err("column diffusion profile must be used on its preparation stream".into());
        }
        let device = Device::Cuda(self.device_index);
        let b = states.batch_size();
        let n = i64::from(self.config.dimension);
        let batch = u32::try_from(b).map_err(|_| "column diffusion batch exceeds CUDA range")?;
        let diagonal = family_a.feature(11).expect("A11 missing diagonal");
        // Both A views come from one packed family allocation. Their batch
        // strides can exceed numel / batch, including when batch == 1.
        let span = |tensor: &Tensor| -> Result<u64, String> {
            let stride = tensor.stride();
            (b - 1)
                .checked_mul(stride[0])
                .and_then(|base| base.checked_add((n - 1).checked_mul(stride[1])?))
                .and_then(|last| last.checked_add(1))
                .and_then(|length| u64::try_from(length).ok())
                .ok_or_else(|| "column diffusion input view overflow".into())
        };
        let diagonal_len = span(diagonal)?;
        for spec in specs
            .iter()
            .filter(|s| s.id.family() == 'G' && s.id.ordinal() <= 4)
        {
            let ordinal = spec.id.ordinal();
            let weight = family_a
                .feature(if ordinal == 2 { 5 } else { 6 })
                .expect("A05/A06 weights");
            let values = Tensor::f_empty([b, n, spec.channels as i64], (Kind::Float, device))
                .map_err(|error| error.to_string())?;
            let status = Tensor::f_empty([b], (Kind::Uint8, device)).map_err(|e| e.to_string())?;
            let args = ffi::Project {
                weight: weight.data_ptr().cast(),
                missing_diagonal: diagonal.data_ptr().cast(),
                descriptors: self.descriptors.data_ptr().cast(),
                profile_status: self.status.data_ptr().cast(),
                values: values.data_ptr().cast(),
                status: status.data_ptr().cast(),
                stream,
                weight_batch_stride: weight.stride()[0] as u64,
                weight_wire_stride: weight.stride()[1] as u64,
                diagonal_batch_stride: diagonal.stride()[0] as u64,
                weight_len: span(weight)?,
                diagonal_len,
                descriptor_len: self.descriptors.numel() as u64,
                profile_status_len: 1,
                value_len: values.numel() as u64,
                status_len: status.numel() as u64,
                batch,
                dimension: self.config.dimension,
                max_n: self.config.max_n,
                ordinal: u32::from(ordinal),
            };
            unsafe { ffi::project(&args)? };
            entries.push(DeviceFeatureEntry::new(
                spec.clone(),
                values,
                status,
                None,
                None,
            ));
        }
        Ok(())
    }
}
