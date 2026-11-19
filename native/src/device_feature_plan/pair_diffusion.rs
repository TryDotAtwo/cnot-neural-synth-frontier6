use crate::cuda::pair_diffusion as ffi;
use crate::device_state::{DeviceFamilyA, DeviceStateBatch};
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{DiffusionConfig, FeatureSpec};
use tch::{Device, Kind, Tensor};

pub(super) struct PairDiffusion {
    config: ffi::Config,
    device_index: usize,
    stream: usize,
    // None means that allocation of a required profile buffer ran out of memory.
    profile: Option<(Tensor, Tensor)>,
}

impl PairDiffusion {
    pub(super) fn new(
        n: u32,
        device_index: usize,
        config: DiffusionConfig,
    ) -> Result<Self, String> {
        let config = ffi::Config {
            dimension: n,
            taus: config.taus,
            floor: config.floor,
            max_pair_states: u64::try_from(config.max_pair_states)
                .map_err(|_| "G06 pair cap overflow")?,
        };
        let layout = ffi::layout(&config)?;
        let c = i64::try_from(layout.classes).map_err(|_| "G06 class count overflow")?;
        let stream = crate::cuda::current_stream(device_index)?;
        let device = Device::Cuda(device_index);
        let empty = |size, kind| Tensor::f_empty([size], (kind, device));
        let storage = (|| -> Result<_, tch::TchError> {
            Ok((
                empty(30 * c, Kind::Double)?,
                empty(16 * c, Kind::Int64)?,
                empty(if c == 0 { 0 } else { i64::from(n) + 1 }, Kind::Double)?,
                empty(4 * i64::from(layout.weight_stride), Kind::Double)?,
                empty(if c == 0 { 0 } else { 5 }, Kind::Uint8)?,
                empty(28 * c, Kind::Float)?,
                empty(1, Kind::Uint8)?,
            ))
        })();
        let mut plan = Self {
            config,
            device_index,
            stream: stream as usize,
            profile: None,
        };
        let (work, pred, factorials, weights, phases, descriptors, status) = match storage {
            Ok(storage) => storage,
            Err(tch::TchError::Torch(message))
                if message.to_ascii_lowercase().contains("out of memory") =>
            {
                return Ok(plan)
            }
            Err(error) => return Err(error.to_string()),
        };
        let args = ffi::Prepare {
            config,
            work: work.data_ptr().cast(),
            predecessors: pred.data_ptr().cast(),
            log_factorial: factorials.data_ptr().cast(),
            weights: weights.data_ptr().cast(),
            phase_status: phases.data_ptr().cast(),
            descriptors: descriptors.data_ptr().cast(),
            status: status.data_ptr().cast(),
            stream,
            work_len: work.numel() as u64,
            predecessor_len: pred.numel() as u64,
            factorial_len: factorials.numel() as u64,
            weight_len: weights.numel() as u64,
            phase_len: phases.numel() as u64,
            descriptor_len: descriptors.numel() as u64,
            status_len: 1,
        };
        unsafe { ffi::prepare(&args)? };
        plan.profile = Some((descriptors, status));
        Ok(plan)
    }

    pub(super) fn project(
        &self,
        states: &DeviceStateBatch,
        family_a: &DeviceFamilyA,
        spec: &FeatureSpec,
    ) -> Result<DeviceFeatureEntry, String> {
        let b = states.batch_size();
        let n = i64::from(self.config.dimension);
        let device = Device::Cuda(self.device_index);
        let stream = crate::cuda::current_stream(self.device_index)?;
        if stream as usize != self.stream {
            return Err("G06 profile must be used on its preparation stream".into());
        }
        let Some((descriptors, profile_status)) = &self.profile else {
            let values =
                Tensor::f_zeros([b, n, n, 28], (Kind::Float, device)).map_err(|e| e.to_string())?;
            let status = Tensor::f_full(
                [b],
                DeviceFeatureStatus::ResourceLimit as i64,
                (Kind::Uint8, device),
            )
            .map_err(|e| e.to_string())?;
            return Ok(DeviceFeatureEntry::new(
                spec.clone(),
                values,
                status,
                None,
                None,
            ));
        };
        let batch = u32::try_from(b).map_err(|_| "G06 batch exceeds CUDA range")?;
        let cells = n * n;
        let pairs = b.checked_mul(cells).ok_or("G06 pair extent overflow")?;
        pairs
            .checked_mul(28 * 4)
            .ok_or("G06 output byte extent overflow")?;
        let class_len = if descriptors.numel() == 0 { 0 } else { pairs };
        let class_ids =
            Tensor::f_empty([class_len], (Kind::Int64, device)).map_err(|e| e.to_string())?;
        let values =
            Tensor::f_empty([b, n, n, 28], (Kind::Float, device)).map_err(|e| e.to_string())?;
        let status = Tensor::f_empty([b], (Kind::Uint8, device)).map_err(|e| e.to_string())?;
        let weight = family_a.feature(6).expect("A06 column weights");
        let intersection = family_a.feature(14).expect("A14 column intersections");
        let span = |tensor: &Tensor, row: i64| {
            (b - 1)
                .checked_mul(tensor.stride()[0])
                .and_then(|base| base.checked_add(row))
                .and_then(|len| u64::try_from(len).ok())
                .ok_or("G06 strided input overflow")
        };
        let args = ffi::Project {
            column_weights: weight.data_ptr().cast(),
            column_intersections: intersection.data_ptr().cast(),
            descriptors: descriptors.data_ptr().cast(),
            profile_status: profile_status.data_ptr().cast(),
            class_ids: class_ids.data_ptr().cast(),
            values: values.data_ptr().cast(),
            status: status.data_ptr().cast(),
            stream,
            max_pair_states: self.config.max_pair_states,
            weight_batch_stride: weight.stride()[0] as u64,
            intersection_batch_stride: intersection.stride()[0] as u64,
            weight_len: span(weight, n)?,
            intersection_len: span(intersection, cells)?,
            descriptor_len: descriptors.numel() as u64,
            profile_status_len: 1,
            class_len: class_ids.numel() as u64,
            value_len: values.numel() as u64,
            status_len: status.numel() as u64,
            batch,
            dimension: self.config.dimension,
        };
        unsafe { ffi::project(&args)? };
        Ok(DeviceFeatureEntry::new(
            spec.clone(),
            values,
            status,
            None,
            None,
        ))
    }
}
