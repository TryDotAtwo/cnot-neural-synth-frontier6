use crate::cuda::shared_word as ffi;
use crate::device_state::DeviceStateBatch;
use crate::model::DeviceFeatureEntry;
use crate::{DiffusionConfig, FeatureSpec};
use tch::{Device, Kind, Tensor};

pub(super) struct SharedWord {
    config: ffi::Config,
    device_index: usize,
    stream: usize,
    transforms: Tensor,
    total_gates: Tensor,
    status: Tensor,
}

impl SharedWord {
    pub(super) fn new(
        n: u32,
        device_index: usize,
        seed: u64,
        config: DiffusionConfig,
    ) -> Result<Self, String> {
        let config = ffi::Config {
            tau: config.mc_tau,
            seed,
            samples: u64::try_from(config.mc_samples)
                .map_err(|_| "shared-word sample count overflow")?,
            dimension: n,
            max_n: config.max_n,
        };
        let words = ffi::layout(&config)?;
        let device = Device::Cuda(device_index);
        let empty = |size, kind| Tensor::f_empty([size], (kind, device)).map_err(|e| e.to_string());
        let transforms = empty(
            i64::try_from(words).map_err(|_| "transform extent overflow")?,
            Kind::Int64,
        )?;
        let total_gates = empty(1, Kind::Int64)?;
        let status = empty(1, Kind::Uint8)?;
        let stream = crate::cuda::current_stream(device_index)?;
        let args = ffi::Prepare {
            config,
            transforms: transforms.data_ptr().cast(),
            total_gates: total_gates.data_ptr().cast(),
            status: status.data_ptr().cast(),
            stream,
            transform_len: words,
            total_len: 1,
            status_len: 1,
        };
        unsafe { ffi::prepare(&args)? };
        Ok(Self {
            config,
            device_index,
            stream: stream as usize,
            transforms,
            total_gates,
            status,
        })
    }

    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        entries: &mut Vec<DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let stream = crate::cuda::current_stream(self.device_index)?;
        if stream as usize != self.stream {
            return Err("shared-word diffusion requires its preparation stream".into());
        }
        let b = states.batch_size();
        let batch = u32::try_from(b).map_err(|_| "shared-word batch exceeds CUDA range")?;
        let n = i64::from(self.config.dimension);
        let device = Device::Cuda(self.device_index);
        let (inputs, sampled, weights) = if self.transforms.numel() == 0 {
            (0, 0, 0)
        } else {
            // The source's working-set guard ensures samples*n² <= 2²².
            let samples = self.config.samples as i64;
            let inputs = n * i64::from(self.config.dimension.div_ceil(32));
            (inputs, inputs * samples, n * samples)
        };
        // Share a fixed scratch allocation across batch chunks in stream order.
        // This scheduling budget does not change the source per-state limit.
        let bytes_per_worker = (inputs + sampled + weights) * 4;
        let workers = if bytes_per_worker == 0 {
            1
        } else {
            b.min(((64 << 20) / bytes_per_worker).max(1))
        };
        let scratch = |size| {
            Tensor::f_empty([workers * size], (Kind::Int, device)).map_err(|e| e.to_string())
        };
        let input_columns = scratch(inputs)?;
        let sample_columns = scratch(sampled)?;
        let sample_weights = scratch(weights)?;
        let mut args = ffi::Eval {
            config: self.config,
            states: states.tensor().data_ptr().cast(),
            transforms: self.transforms.data_ptr().cast(),
            total_gates: self.total_gates.data_ptr().cast(),
            profile_status: self.status.data_ptr().cast(),
            input_columns: input_columns.data_ptr().cast(),
            sample_columns: sample_columns.data_ptr().cast(),
            weights: sample_weights.data_ptr().cast(),
            values: [std::ptr::null_mut(); 4],
            statuses: [std::ptr::null_mut(); 4],
            stream,
            state_len: states.tensor().numel() as u64,
            transform_len: self.transforms.numel() as u64,
            total_len: 1,
            profile_status_len: 1,
            input_len: input_columns.numel() as u64,
            sample_len: sample_columns.numel() as u64,
            weight_len: sample_weights.numel() as u64,
            value_len: [0; 4],
            status_len: [0; 4],
            batch,
            workers: workers as u32,
            selected: 0,
        };
        let mut outputs = Vec::with_capacity(4);
        for spec in specs
            .iter()
            .filter(|s| s.id.family() == 'G' && (7..=10).contains(&s.id.ordinal()))
        {
            let slot = usize::from(spec.id.ordinal() - 7);
            let channels = spec.channels as i64;
            let shape = if slot == 0 {
                vec![b, n, channels]
            } else {
                vec![b, n, n, channels]
            };
            shape
                .iter()
                .try_fold(4_i64, |size, &axis| size.checked_mul(axis))
                .ok_or("shared-word output byte extent overflow")?;
            let values = Tensor::f_empty(shape.as_slice(), (Kind::Float, device))
                .map_err(|e| e.to_string())?;
            let status = Tensor::f_empty([b], (Kind::Uint8, device)).map_err(|e| e.to_string())?;
            args.selected |= 1 << slot;
            args.values[slot] = values.data_ptr().cast();
            args.statuses[slot] = status.data_ptr().cast();
            args.value_len[slot] = values.numel() as u64;
            args.status_len[slot] = status.numel() as u64;
            outputs.push(DeviceFeatureEntry::new(
                spec.clone(),
                values,
                status,
                None,
                None,
            ));
        }
        unsafe { ffi::evaluate(&args)? };
        entries.extend(outputs);
        Ok(())
    }
}
